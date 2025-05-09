"""
This module contains the routine to initialize quantities like
curvature and torsion from the magnetix axis shape.
"""

import logging
import numpy as np
import torch
from scipy.interpolate import CubicSpline as spline
from .spectral_diff_matrix import spectral_diff_matrix
from .util import fourier_minimum

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define periodic spline interpolant conversion used in several scripts and plotting
def convert_to_spline(self, tensor):
    # TODO: this is still using a scipy spline; convert it to torch
    x = torch.cat((self.phi, torch.tensor([2 * torch.pi / self.nfp])))
    y = torch.cat((tensor, torch.tensor([tensor[0]])))
    sp = spline(x.detach().numpy(), y.detach().numpy(), bc_type='periodic')
    return sp


def init_axis(self):
    """
    Initialize the curvature, torsion, differentiation matrix, etc.
    """
    # Shorthand:
    nphi = self.nphi
    nfp = self.nfp

    # dont solve the vacuum case unless we are in vacuum mode
    self.solve_vacuum = False

    phi = torch.tensor(np.linspace(0, 2 * torch.pi / nfp, nphi, endpoint=False))
    d_phi = phi[1] - phi[0]
    R0 = torch.zeros(nphi)
    Z0 = torch.zeros(nphi)
    R0p = torch.zeros(nphi)
    Z0p = torch.zeros(nphi)
    R0pp = torch.zeros(nphi)
    Z0pp = torch.zeros(nphi)
    R0ppp = torch.zeros(nphi)
    Z0ppp = torch.zeros(nphi)
    for jn in range(0, self.nfourier):
        n = jn * nfp
        sinangle = torch.sin(n * phi)
        cosangle = torch.cos(n * phi)
        R0 += self.rc[jn] * cosangle + self.rs[jn] * sinangle
        Z0 += self.zc[jn] * cosangle + self.zs[jn] * sinangle
        R0p += self.rc[jn] * (-n * sinangle) + self.rs[jn] * (n * cosangle)
        Z0p += self.zc[jn] * (-n * sinangle) + self.zs[jn] * (n * cosangle)
        R0pp += self.rc[jn] * (-n * n * cosangle) + self.rs[jn] * (-n * n * sinangle)
        Z0pp += self.zc[jn] * (-n * n * cosangle) + self.zs[jn] * (-n * n * sinangle)
        R0ppp += self.rc[jn] * (n * n * n * sinangle) + self.rs[jn] * (-n * n * n * cosangle)
        Z0ppp += self.zc[jn] * (n * n * n * sinangle) + self.zs[jn] * (-n * n * n * cosangle)

    d_l_d_phi = torch.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
    d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
    B0_over_abs_G0 = nphi / torch.sum(d_l_d_phi)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    self.d_l_d_varphi = abs_G0_over_B0
    G0 = self.sG * abs_G0_over_B0 * self.B0

    # For these next arrays, the first dimension is phi, and the 2nd dimension is (R, phi, Z).
    d_r_d_phi_cylindrical = torch.stack([R0p, R0, Z0p], dim=1)
    d2_r_d_phi2_cylindrical = torch.stack([R0pp - R0, 2 * R0p, Z0pp], dim=1)
    d3_r_d_phi3_cylindrical = torch.stack([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp], dim=1)

    tangent_cylindrical = torch.zeros((nphi, 3))
    d_tangent_d_l_cylindrical = torch.zeros((nphi, 3))
    for j in range(3):
        tangent_cylindrical[:,j] = d_r_d_phi_cylindrical[:,j] / d_l_d_phi
        d_tangent_d_l_cylindrical[:,j] = (-d_r_d_phi_cylindrical[:,j] * d2_l_d_phi2 / d_l_d_phi \
                                          + d2_r_d_phi2_cylindrical[:,j]) / (d_l_d_phi * d_l_d_phi)

    curvature = torch.sqrt(d_tangent_d_l_cylindrical[:,0] * d_tangent_d_l_cylindrical[:,0] + \
                        d_tangent_d_l_cylindrical[:,1] * d_tangent_d_l_cylindrical[:,1] + \
                        d_tangent_d_l_cylindrical[:,2] * d_tangent_d_l_cylindrical[:,2])

    axis_length = torch.sum(d_l_d_phi) * d_phi * nfp
    rms_curvature = torch.sqrt((torch.sum(curvature * curvature * d_l_d_phi) * d_phi * nfp) / axis_length)
    mean_of_R = torch.sum(R0 * d_l_d_phi) * d_phi * nfp / axis_length
    mean_of_Z = torch.sum(Z0 * d_l_d_phi) * d_phi * nfp / axis_length
    standard_deviation_of_R = torch.sqrt(torch.sum((R0 - mean_of_R) ** 2 * d_l_d_phi) * d_phi * nfp / axis_length)
    standard_deviation_of_Z = torch.sqrt(torch.sum((Z0 - mean_of_Z) ** 2 * d_l_d_phi) * d_phi * nfp / axis_length)

    normal_cylindrical = torch.zeros((nphi, 3))
    for j in range(3):
        normal_cylindrical[:,j] = d_tangent_d_l_cylindrical[:,j] / curvature
    self.normal_cylindrical = normal_cylindrical
    self._determine_helicity()

    # b = t x n
    binormal_cylindrical = torch.zeros((nphi, 3))
    binormal_cylindrical[:,0] = tangent_cylindrical[:,1] * normal_cylindrical[:,2] - tangent_cylindrical[:,2] * normal_cylindrical[:,1]
    binormal_cylindrical[:,1] = tangent_cylindrical[:,2] * normal_cylindrical[:,0] - tangent_cylindrical[:,0] * normal_cylindrical[:,2]
    binormal_cylindrical[:,2] = tangent_cylindrical[:,0] * normal_cylindrical[:,1] - tangent_cylindrical[:,1] * normal_cylindrical[:,0]

    # We use the same sign convention for torsion as the
    # Landreman-Sengupta-Plunk paper, wikipedia, and
    # mathworld.wolfram.com/Torsion.html.  This sign convention is
    # opposite to Garren & Boozer's sign convention!
    torsion_numerator = (d_r_d_phi_cylindrical[:,0] * (d2_r_d_phi2_cylindrical[:,1] * d3_r_d_phi3_cylindrical[:,2] - d2_r_d_phi2_cylindrical[:,2] * d3_r_d_phi3_cylindrical[:,1]) \
                         + d_r_d_phi_cylindrical[:,1] * (d2_r_d_phi2_cylindrical[:,2] * d3_r_d_phi3_cylindrical[:,0] - d2_r_d_phi2_cylindrical[:,0] * d3_r_d_phi3_cylindrical[:,2]) 
                         + d_r_d_phi_cylindrical[:,2] * (d2_r_d_phi2_cylindrical[:,0] * d3_r_d_phi3_cylindrical[:,1] - d2_r_d_phi2_cylindrical[:,1] * d3_r_d_phi3_cylindrical[:,0]))

    torsion_denominator = (d_r_d_phi_cylindrical[:,1] * d2_r_d_phi2_cylindrical[:,2] - d_r_d_phi_cylindrical[:,2] * d2_r_d_phi2_cylindrical[:,1]) ** 2 \
        + (d_r_d_phi_cylindrical[:,2] * d2_r_d_phi2_cylindrical[:,0] - d_r_d_phi_cylindrical[:,0] * d2_r_d_phi2_cylindrical[:,2]) ** 2 \
        + (d_r_d_phi_cylindrical[:,0] * d2_r_d_phi2_cylindrical[:,1] - d_r_d_phi_cylindrical[:,1] * d2_r_d_phi2_cylindrical[:,0]) ** 2

    torsion = torsion_numerator / torsion_denominator

    self.etabar_squared_over_curvature_squared = self.etabar * self.etabar / (curvature * curvature)

    self.d_d_phi = spectral_diff_matrix(self.nphi, xmax=2 * torch.pi / self.nfp)
    self.d_varphi_d_phi = B0_over_abs_G0 * d_l_d_phi
    self.d_d_varphi = torch.zeros((nphi, nphi))
    for j in range(nphi):
        self.d_d_varphi[j,:] = self.d_d_phi[j,:] / self.d_varphi_d_phi[j]

    # TODO: integrate this with scipy to see if it is dominating the error
    # Compute the Boozer toroidal angle:
    self.varphi = torch.zeros(nphi)
    for j in range(1, nphi):
        # To get toroidal angle on the full mesh, we need d_l_d_phi on the half mesh.
        self.varphi[j] = self.varphi[j-1] + (d_l_d_phi[j-1] + d_l_d_phi[j])
    self.varphi = self.varphi * (0.5 * d_phi * 2 * torch.pi / axis_length)

    # Cartesian coordinates of axis
    self.XYZ0 = torch.stack((R0 * torch.cos(phi), R0 * torch.sin(phi), Z0)) # (3, nphi)
    # derivative of axis wrt phi
    X0p = R0p * torch.cos(phi) - R0 * torch.sin(phi)
    Y0p = R0p * torch.sin(phi) + R0 * torch.cos(phi)
    self.dXYZ0_by_dphi = torch.stack([X0p, Y0p, Z0p]) # (3, nphi)

    # Add all results to self:
    self.phi = phi
    self.d_phi = d_phi
    self.R0 = R0
    self.Z0 = Z0
    self.R0p = R0p
    self.X0p = X0p
    self.Y0p = Y0p
    self.Z0p = Z0p
    self.R0pp = R0pp
    self.Z0pp = Z0pp
    self.R0ppp = R0ppp
    self.Z0ppp = Z0ppp
    self.G0 = G0
    self.d_l_d_phi = d_l_d_phi
    self.axis_length = axis_length
    self.curvature = curvature
    self.torsion = torsion
    self.X1s = torch.zeros(nphi)
    self.X1c = self.etabar / curvature

    # TODO: fourier_minimum is not in torch yet
    self.min_R0 = fourier_minimum(self.R0.detach().numpy())

    self.tangent_cylindrical = tangent_cylindrical
    self.normal_cylindrical = normal_cylindrical 
    self.binormal_cylindrical = binormal_cylindrical
    self.Bbar = self.spsi * self.B0
    self.abs_G0_over_B0 = abs_G0_over_B0

    # TODO: unit test the cartesian frenet-frame
    cosphi = torch.cos(phi)
    sinphi = torch.sin(phi)
    # n_x = n_R * cos(phi) - n_phi * sin(phi)
    normal_x = normal_cylindrical[:,0] * cosphi - normal_cylindrical[:,1] * sinphi
    # n_y= n_R * sin(phi) + n_phi * cos(phi)
    normal_y = normal_cylindrical[:,0] * sinphi + normal_cylindrical[:,1] * cosphi
    binormal_x = binormal_cylindrical[:,0] * cosphi - binormal_cylindrical[:,1] * sinphi
    binormal_y = binormal_cylindrical[:,0] * sinphi + binormal_cylindrical[:,1] * cosphi
    tangent_x = tangent_cylindrical[:,0] * cosphi - tangent_cylindrical[:,1] * sinphi
    tangent_y = tangent_cylindrical[:,0] * sinphi + tangent_cylindrical[:,1] * cosphi

    self.normal_cartesian = torch.stack([normal_x, normal_y, normal_cylindrical[:,2]]).T # (nphi, 3)
    self.binormal_cartesian = torch.stack([binormal_x, binormal_y, binormal_cylindrical[:,2]]).T # (nphi, 3)
    self.tangent_cartesian = torch.stack([tangent_x, tangent_y, tangent_cylindrical[:,2]]).T # (nphi, 3)
    
    # The output is not stellarator-symmetric if (1) R0s is nonzero,
    # (2) Z0c is nonzero, (3) sigma_initial is nonzero, or (B2s is
    # nonzero and order != 'r1')
    self.lasym = torch.max(torch.abs(self.rs)) > 0 or torch.max(torch.abs(self.zc)) > 0 \
        or self.sigma0 != 0 or (self.order != 'r1' and self.B2s != 0)

    # TODO: use torch interpolation here
    # Functions that converts a toroidal angle phi0 on the axis to the axis radial and vertical coordinates
    self.R0_func = self.convert_to_spline(sum([self.rc[i]*torch.cos(i*self.nfp*self.phi) +\
                                               self.rs[i]*torch.sin(i*self.nfp*self.phi) \
                                              for i in range(len(self.rc))]))
    self.Z0_func = self.convert_to_spline(sum([self.zc[i]*torch.cos(i*self.nfp*self.phi) +\
                                               self.zs[i]*torch.sin(i*self.nfp*self.phi) \
                                              for i in range(len(self.zs))]))

    # Spline interpolants for the cylindrical components of the Frenet-Serret frame:
    self.normal_R_spline     = self.convert_to_spline(self.normal_cylindrical[:,0])
    self.normal_phi_spline   = self.convert_to_spline(self.normal_cylindrical[:,1])
    self.normal_z_spline     = self.convert_to_spline(self.normal_cylindrical[:,2])
    self.binormal_R_spline   = self.convert_to_spline(self.binormal_cylindrical[:,0])
    self.binormal_phi_spline = self.convert_to_spline(self.binormal_cylindrical[:,1])
    self.binormal_z_spline   = self.convert_to_spline(self.binormal_cylindrical[:,2])
    self.tangent_R_spline    = self.convert_to_spline(self.tangent_cylindrical[:,0])
    self.tangent_phi_spline  = self.convert_to_spline(self.tangent_cylindrical[:,1])
    self.tangent_z_spline    = self.convert_to_spline(self.tangent_cylindrical[:,2])

    # Spline interpolant for nu = varphi - phi, used for plotting
    self.nu_spline = self.convert_to_spline(self.varphi - self.phi)
