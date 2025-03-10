#!/usr/bin/env python3

"""
Methods for computing the virtual casing integral.
"""

import logging
import numpy as np
import torch
from .util import rotate_nfp

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def B_taylor(self, r, ntheta=64):
    """Calculate the magnetic field on a flux surface of radius r using
    the Taylor expansion of B.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 64.

    Returns:
        B_surf: (nphi, ntheta, 3) array containing the magnetic field on the flux surface.
    """

    self.calculate()
    B0 = self.Bfield_cartesian() # (3, nphi)

    # [i, j, k]; i indexes Cartesian dofs; j indexes B; k indexes axis
    gradB = self.grad_B_tensor_cartesian() # (3, 3, nphi)
    if self.order != 'r1':
        # [i, j, k, l]; k indexes B, (i,j) are Cartesian dofs; l indexes axis.
        grad2B = self.grad_grad_B_tensor_cartesian() # (3, 3, 3, nphi)
    nphi = self.nphi

    # compute flux surface
    gamma_axis = self.XYZ0.T # (nphi, 3)
    gamma_surf = self.surface(r, ntheta=ntheta) # (nphi, ntheta, 3)

    # now Taylor expand
    delta_r = gamma_surf - gamma_axis.reshape((-1,1,3)) # (nphi, ntheta, 3)

    B_surf = torch.zeros((nphi, ntheta, 3))
    for ii in range(nphi):

        B_surf[ii] = B0[:,ii] + torch.einsum('ij,ki->kj', gradB[:,:,ii], delta_r[ii]) # (ntheta, 3)

        if self.order != 'r1':
            part = torch.einsum('ijk,li->kjl', grad2B[:,:,:,ii], delta_r[ii]) # (3, 3, ntheta)
            B_surf[ii] += 0.5 * torch.einsum('lj,kjl->lk', delta_r[ii], part) # (ntheta, 3)

    return B_surf

def B_external_on_axis(self, r=0.1, ntheta=128, X_target=[]):
    """Compute B_external on the magnetic axis using the virtual casing principle.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 64.
        X_target (tensor, optional): (n, 3) tensor of n target points inside the surface
            of radius r at which to evaluate B_external. The points do not necessarily need
            to be on magnetic axis. Defaults to (nphi, 3) tensor of points on the magnetic 
            axis, uniformly spaced in the axis cylindrical phi.
    Returns:
        (tensor): (3, n) tensor of evaluations of B_external.
    """
    if len(X_target) == 0:
        X_target = self.XYZ0.T # (nphi, 3)
    n_target = len(X_target)

    I = 0.0
    G = self.G0
    if self.order != 'r1':
        I += r**2 * self.I2
        G += r**2 * self.G2

    g = self.surface(r=r, ntheta=ntheta) # (nphi, ntheta, 3)
    gtheta = self.dsurface_by_dtheta(r=r, ntheta=ntheta) # (nphi, ntheta, 3)
    gphi = self.dsurface_by_dvarphi(r=r, ntheta=ntheta) # (nphi, ntheta, 3)

    # get surface and tangents across all nfp
    dr_by_dtheta = torch.zeros((int(self.nfp * self.nphi), ntheta, 3))
    dr_by_dvarphi = torch.zeros((int(self.nfp * self.nphi), ntheta, 3))
    gamma_surf = torch.zeros((int(self.nfp * self.nphi), ntheta, 3))
    for ii in range(self.nfp):
        g = rotate_nfp(g, ii, self.nfp)
        gtheta = rotate_nfp(gtheta, ii, self.nfp)
        gphi = rotate_nfp(gphi, ii, self.nfp)
        gamma_surf[ii * self.nphi : (ii+1) * self.nphi] = g
        dr_by_dtheta[ii * self.nphi : (ii+1) * self.nphi] = gtheta
        dr_by_dvarphi[ii * self.nphi : (ii+1) * self.nphi] = gphi

    dtheta = 2 * torch.pi / ntheta
    dphi = torch.diff(self.phi)[0]
    dvarphi_by_dphi = self.d_varphi_d_phi
    dvarphi_by_dphi = torch.concatenate([dvarphi_by_dphi for ii in range(self.nfp)]).flatten().reshape((-1,1,1))

    def B_ext_of_phi(ii):
        """ Compute B_external by integrating over the entire device. """

        # biot-savart kernel
        rprime = X_target[ii] - gamma_surf # (nphi, ntheta, 3)
        norm_rprime_cubed = (torch.sqrt(torch.sum(rprime**2, dim=-1, keepdims=True))**3) # (nphi, ntheta, 1)
        kernel = rprime / norm_rprime_cubed

        # cross product
        diff = I * dr_by_dvarphi - G * dr_by_dtheta
        integrand = torch.linalg.cross(kernel, diff, dim=-1) # (nphi, ntheta, 3)

        integral =  (1.0 / (4 * torch.pi) ) * torch.sum(integrand * dvarphi_by_dphi *  dtheta * dphi, dim=(0,1)) # (3,)

        return integral
    
    B_ext = torch.stack([B_ext_of_phi(ii) for ii in range(n_target)]).T

    return B_ext

def B_external_on_axis_taylor(self, r=0.1, ntheta=128, X_target=[]):
    """Compute B_external on the magnetic axis using the virtual casing principle.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 64.
        X_target (tensor, optional): (n, 3) tensor of n target points inside the surface
            of radius r at which to evaluate B_external. The points do not necessarily need
            to be on magnetic axis. Defaults to (nphi, 3) tensor of points on the magnetic 
            axis, uniformly spaced in the axis cylindrical phi.
    Returns:
        (tensor): (3, n) tensor of evaluations of B_external.
    """
    if len(X_target) == 0:
        X_target = self.XYZ0.T # (nphi, 3)

    I = 0.0
    G = self.G0
    if self.order != 'r1':
        I += r**2 * self.I2
        G += r**2 * self.G2

    n = self.surface_normal(r=r, ntheta=ntheta) # (nphi, ntheta, 3)
    g = self.surface(r=r, ntheta=ntheta) # (nphi, ntheta, 3)
    b = self.B_taylor(r=r, ntheta=ntheta) # (nphi, ntheta, 3)

    # get the surface and normal across all nfp
    normal = torch.zeros((int(self.nfp * self.nphi), ntheta, 3))
    gamma_surf = torch.zeros((int(self.nfp * self.nphi), ntheta, 3))
    B = torch.zeros((int(self.nfp * self.nphi), ntheta, 3))
    for ii in range(self.nfp):
        n = rotate_nfp(n, ii, self.nfp)
        g = rotate_nfp(g, ii, self.nfp)
        b = rotate_nfp(b, ii, self.nfp)
        normal[ii * self.nphi : (ii+1) * self.nphi] = n
        gamma_surf[ii * self.nphi : (ii+1) * self.nphi] = g
        B[ii * self.nphi : (ii+1) * self.nphi] = b

    dtheta = 2 * torch.pi / ntheta
    dphi = torch.diff(self.phi)[0]
    dvarphi_by_dphi = self.d_varphi_d_phi
    dvarphi_by_dphi = torch.concatenate([dvarphi_by_dphi for ii in range(self.nfp)]).flatten().reshape((-1,1,1))

    # return B_ext
    dA = torch.sum(normal**2, dim=-1, keepdims=True)

    # TODO: check the direction of the normal
    # use outward facing unit normal
    nhat = normal / dA

    B_ext = torch.zeros(len(X_target), 3)
    
    for ii, xx in enumerate(X_target):

        """Compute B_external(phi) by integrating over surface."""

        rprime = xx - gamma_surf # (nphi, ntheta, 3)

        # a = x - y / |x - y|^3 (y is surface point)
        norm_rprime_cubed = (torch.sqrt(torch.sum(rprime**2, dim=-1, keepdims=True))**3) # (nphi, ntheta, 1)
        kernel = rprime / norm_rprime_cubed

        n_cross_B = torch.linalg.cross(nhat, B, dim=-1) # (nphi, ntheta, 3)
        k_cross_n_cross_B = torch.linalg.cross(kernel, n_cross_B, dim=-1) # (nphi, ntheta, 3)
        
        # # dot product term
        # n_dot_B = torch.sum(nhat * B, dim=-1, keepdims=True) # (nphi, ntheta, 1)
        # k_n_dot_B = kernel * n_dot_B  # (nphi, ntheta, 3)

        integrand = k_cross_n_cross_B
        # integrate
        B_ext[ii] =  (1.0 / (4 * torch.pi) ) * torch.sum(integrand * dA * dvarphi_by_dphi * dtheta * dphi, dim=(0,1)) # (3,)

    return B_ext.T