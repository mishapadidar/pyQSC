#!/usr/bin/env python3

"""
Methods for computing the virtual casing integral.
"""

import logging
import numpy as np
from functools import lru_cache
import torch
from .util import rotate_nfp
from .fourier_tools import fourier_interp1d
from torch.special import erf


#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def divergence_taylor(self, r, ntheta=64, vacuum_component=False):
    """Compute the divergence of the Taylor expansion of B on a flux surface.
    
    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 64.
        vacuum_component (bool, optional): If true, only compute the divergence of the vacuum component of the
            field. Defaults to False.

    Returns:
        (tensor): (nphi, ntheta) tensor of evaluations of the divergence of B.
    """
    # [i, j, k]; i indexes Cartesian dofs; j indexes B; k indexes axis
    gradB = self.grad_B_tensor_cartesian(vacuum_component=vacuum_component) # (3, 3, nphi)

    # divergence of first order expansion
    divB = gradB.diagonal(offset=0, dim1=0, dim2=1).sum(dim=-1) # (nphi,)
    divB = torch.tile(divB, (ntheta,1)).T # (nphi, ntheta)

    if self.order != 'r1':
        # [i, j, k, l]; k indexes B, (i,j) are Cartesian dofs; l indexes axis.
        grad2B = self.grad_grad_B_tensor_cartesian(vacuum_component=vacuum_component) # (3, 3, 3, nphi)

        # compute flux surface
        gamma_axis = torch.clone(self.XYZ0.T) # (nphi, 3)
        gamma_surf = self.surface(r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
        delta_r = gamma_surf - gamma_axis.reshape((-1,1,3)) # (nphi, ntheta, 3)
        
        for ll in range(self.nphi):
            # compute divB
            Aii = (grad2B[:,0,0,ll] + grad2B[:,1,1,ll] + grad2B[:,2,2,ll]) # (3,)          
            divB[ll] += torch.matmul(delta_r[ll], Aii) # (ntheta,)

    return divB


def curl_taylor(self, r, ntheta=64, vacuum_component=False):
    """Compute the curl of the Taylor expansion of B on a flux surface.
    
    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 64.
        vacuum_component (bool, optional): If true, only compute the curl of the vacuum component of the
            field. Defaults to False.

    Returns:
        (tensor): (nphi, ntheta, 3) tensor of evaluations of the curl of B.
    """
    # [i, j, k]; i indexes Cartesian dofs; j indexes B; k indexes axis
    gradB = self.grad_B_tensor_cartesian(vacuum_component=vacuum_component) # (3, 3, nphi)

    # storage
    curlB = torch.zeros((self.nphi, ntheta, 3)) # (nphi, ntheta, 3)

    # curl of first order expansion
    A32 = gradB[1, 2, :] # (nphi,)
    A23 = gradB[2, 1, :] # (nphi,)
    A13 = gradB[2, 0, :] # (nphi,)
    A31 = gradB[0, 2, :] # (nphi,)
    A21 = gradB[0, 1, :] # (nphi,)
    A12 = gradB[1, 0, :] # (nphi,)
    curlB[:, :, 0] = (A32 - A23)[:, None]
    curlB[:, :, 1] = (A13 - A31)[:, None]
    curlB[:, :, 2] = (A21 - A12)[:, None]

    if self.order != 'r1':
        # [i, j, k, l]; k indexes B, (i,j) are Cartesian dofs; l indexes axis.
        grad2B = self.grad_grad_B_tensor_cartesian(vacuum_component=vacuum_component) # (3, 3, 3, nphi)

        # compute flux surface
        gamma_axis = torch.clone(self.XYZ0.T) # (nphi, 3)
        gamma_surf = self.surface(r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
        delta_r = gamma_surf - gamma_axis.reshape((-1,1,3)) # (nphi, ntheta, 3)
        
        for ll in range(self.nphi):
            
            # compute curlB
            A32 = grad2B[:, 1, 2, ll] # (3,)
            A23 = grad2B[:, 2, 1, ll] # (3,)
            A13 = grad2B[:, 2, 0, ll] # (3,)
            A31 = grad2B[:, 0, 2, ll] # (3,)
            A21 = grad2B[:, 0, 1, ll] # (3,)
            A12 = grad2B[:, 1, 0, ll] # (3,)
            grad2B_antisymmetric = torch.zeros((3, 3)) # (3, 3)
            grad2B_antisymmetric[0] = A32 - A23
            grad2B_antisymmetric[1] = A13 - A31
            grad2B_antisymmetric[2] = A21 - A12
            
            # curl of the second order expansion
            curlB[ll] += torch.matmul(grad2B_antisymmetric, delta_r[ll].T).T # (ntheta, 3)

    return curlB


def B_taylor(self, r, ntheta=64, vacuum_component=False):
    """Calculate the magnetic field on a flux surface of radius r using
    the Taylor expansion of B.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 64.
        vacuum_component (bool, optional): If true, only compute the vacuum component of the
            field on the vacuum surface. Defaults to False.

    Returns:
        B_surf: (nphi, ntheta, 3) array containing the magnetic field on the flux surface.
    """

    B0 = self.Bfield_cartesian() # (3, nphi)

    # [i, j, k]; i indexes Cartesian dofs; j indexes B; k indexes axis
    gradB = self.grad_B_tensor_cartesian(vacuum_component=vacuum_component) # (3, 3, nphi)
    if self.order != 'r1':
        # [i, j, k, l]; k indexes B, (i,j) are Cartesian dofs; l indexes axis.
        grad2B = self.grad_grad_B_tensor_cartesian(vacuum_component=vacuum_component) # (3, 3, 3, nphi)
    nphi = self.nphi

    # compute flux surface
    gamma_axis = torch.clone(self.XYZ0.T) # (nphi, 3)
    gamma_surf = self.surface(r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)

    # now Taylor expand
    delta_r = gamma_surf - gamma_axis.reshape((-1,1,3)) # (nphi, ntheta, 3)

    B_surf = torch.zeros((nphi, ntheta, 3))
    for ii in range(nphi):

        B_surf[ii] = B0[:,ii] + torch.einsum('ij,ki->kj', gradB[:,:,ii], delta_r[ii]) # (ntheta, 3)

        if self.order != 'r1':
            part = torch.einsum('ijk,li->kjl', grad2B[:,:,:,ii], delta_r[ii]) # (3, 3, ntheta)
            B_surf[ii] += 0.5 * torch.einsum('lj,kjl->lk', delta_r[ii], part) # (ntheta, 3)

    return B_surf

def B_external_on_axis(self, r=0.1, ntheta=256, nphi=1024):
    """Compute B_external on the magnetic axis using the virtual casing principle. If the
    configuration is in vacuum, the vacuum solution is returned.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.

    Returns:
        (tensor): (3, n) tensor of evaluations of B_external.
    """

    if (self.p2 == 0.0) and (self.I2 == 0.0):
        # in vacuum, return vacuum solution
        return self.Bfield_cartesian()
    else:
        return self.B_external_on_axis_taylor(r=r, ntheta=ntheta, nphi=nphi)
        # return self.B_external_on_axis_corrected(r=r, ntheta=ntheta, nphi=nphi)
    
def grad_B_external_on_axis(self, r=0.1, ntheta=256, nphi=1024):
    """Compute grad_B_external on the magnetic axis using the virtual casing principle. If the
    configuration is in vacuum, the vacuum solution is returned.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.

    Returns:
        (tensor): (3, 3, n) tensor of evaluations of grad_B_external.
    """

    if (self.p2 == 0.0) and (self.I2 == 0.0):
        # in vacuum, return vacuum solution
        return self.grad_B_tensor_cartesian()
    else:
        return self.grad_B_external_on_axis_taylor(r=r, ntheta=ntheta, nphi=nphi)
        # return self.grad_B_external_on_axis_corrected(r=r, ntheta=ntheta, nphi=nphi)
        
@lru_cache(maxsize=8)
def B_external_on_axis_taylor(self, r=0.1, ntheta=256, nphi=1024, X_target=[], vacuum_component=False):
    """Compute B_external on the magnetic axis using the virtual casing principle,
        Bext(r) = (1/4pi) int k(r,r') x (n(r') x B(r')) dtheta dphi
    If vacuum_component is True, then the integral is computed assuming p2=I2=0,
        Bext_vac(r) = (1/4pi) int k(r,r_vac') x (n_vac(r') x B_vac(r')) dtheta dphi
    where r_vac, n_vac and B_vac are the flux surface, surface normal and magnetic field
    computed assuming vacuum (p2=I2=0).

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
        X_target (tensor, optional): (n, 3) tensor of n target points inside the surface
            of radius r at which to evaluate B_external. The points do not necessarily need
            to be on magnetic axis. Defaults to (nphi, 3) tensor of points on the magnetic 
            axis, uniformly spaced in the axis cylindrical phi.
        vacuum_component (bool, optional): If true, computes the integral assuming (p2=I2=0).
            Defaults to False.
    Returns:
        (tensor): (3, n) tensor of evaluations of B_external.
    """
    # TODO: rename this method since target points need not be on axis
    if len(X_target) == 0:
        X_target = torch.clone(self.XYZ0.T) # (nphi, 3)
    n_target = len(X_target)

    # # interpolate
    n_cross_B_interp, gamma_surf_interp = build_virtual_casing_interpolants(self, r=r, ntheta=ntheta, nphi=nphi, vacuum_component=vacuum_component)

    dtheta = 2 * torch.pi / ntheta
    dphi = 2 * torch.pi / nphi

    @torch.compile    
    def B_ext_of_phi(ii):
        """ Compute B_external by integrating over the entire device. """

        # biot-savart kernel
        rprime = X_target[ii] - gamma_surf_interp # (nphi, ntheta, 3)
        norm_rprime_cubed = (torch.sqrt(torch.sum(rprime**2, dim=-1, keepdims=True))**3) # (nphi, ntheta, 1)
        kernel = rprime / norm_rprime_cubed

        # cross product
        integrand = torch.linalg.cross(kernel, n_cross_B_interp, dim=-1) # (nphi, ntheta, 3)

        integral =  (1.0 / (4 * torch.pi) ) * torch.sum(integrand * dtheta * dphi, dim=(0,1)) # (3,)

        return integral
    
    B_ext = torch.stack([B_ext_of_phi(ii) for ii in range(n_target)]).T

    return B_ext

@lru_cache(maxsize=8)
def B_external_on_axis_taylor_singularity_subtraction(self, r=0.1, ntheta=256, n_intervals=30, vacuum_component=False):
    """Compute B_external on the magnetic axis using the virtual casing principle,
        Bext(r) = (1/4pi) int k(r,r') x (n(r') x B(r')) dtheta dphi
    If vacuum_component is True, then the integral is computed assuming p2=I2=0,
        Bext_vac(r) = (1/4pi) int k(r,r_vac') x (n_vac(r') x B_vac(r')) dtheta dphi
    where r_vac, n_vac and B_vac are the flux surface, surface normal and magnetic field
    computed assuming vacuum (p2=I2=0).

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
        X_target (tensor, optional): (n, 3) tensor of n target points inside the surface
            of radius r at which to evaluate B_external. The points do not necessarily need
            to be on magnetic axis. Defaults to (nphi, 3) tensor of points on the magnetic 
            axis, uniformly spaced in the axis cylindrical phi.
        vacuum_component (bool, optional): If true, computes the integral assuming (p2=I2=0).
            Defaults to False.
    Returns:
        (tensor): (3, n) tensor of evaluations of B_external.
    """
    X_target = torch.clone(self.XYZ0.T) # (nphi, 3)
    n_target = len(X_target)

    # period of most functions is now 2pi
    period = 2 * torch.pi

    # window function params
    """
    The kernel decays like k ~ 1/distance^2. Distance across a chord of a circle is approximately 2 * R * sin(theta/2).
    Pythagorean theorem gives us 
        distance^2 = (2 * R * sin(theta/2))^2 + (r)^2,
    so 
        k ~ 1 / [ (2 * R * sin(theta/2))^2 + (r)^2 ]
    For the singularity to no longer dominate, we want to be at a distance where
        (2 * R * sin(theta/2))^2  ~ (r)^2
    => sin(theta/2) ~ r / (2 * R)
    => theta ~ 2 * arcsin(r / (2 * R))
    which is the half-width of the window function.
    We increase it to account for the fact that the surface may be elongated.
    """
    sharpness = 30
    aspect_ratio = self.rc[0].detach().numpy().item() / r
    half_width = 2 * np.arcsin(0.5 / aspect_ratio)
    def window_func(phip, center, sharpness, half_width):
        return 0.5 * (erf(sharpness*((phip - center) + half_width)) - erf(sharpness*((phip - center) - half_width)))
    # test window function decays fast enough
    np.testing.assert_allclose(window_func(torch.tensor(0.0 + period/2), 0.0, sharpness, half_width), 0.0, atol=1e-7)
    np.testing.assert_allclose(window_func(torch.tensor(0.0 - period/2), 0.0, sharpness, half_width), 0.0, atol=1e-7)
    def periodic_window_func(phip, center, sharpness, half_width):
        # an approximately periodic window function
        # it is periodic to numerical precisision if window_func(center + period/2, center, sharpness, half_width)=0
        return window_func(phip, center, sharpness, half_width) + window_func(phip + period, center, sharpness, half_width) + window_func(phip - period, center, sharpness, half_width)
    
    phi_target = self.phi # phi values of target points

    # data for the integrand
    n_cross_B, gamma_surf = build_virtual_casing_grid(self, r=r, ntheta=ntheta, vacuum_component=vacuum_component)

    # interpolate
    nphi_smooth = n_cross_B.shape[0]
    phi_smooth = torch.linspace(0, 2*torch.pi, nphi_smooth+1)[:-1] # phi values of quad points
    # n_cross_B_smooth, gamma_surf_smooth = build_virtual_casing_interpolants(self, r=r, ntheta=ntheta, nphi=nphi_smooth, vacuum_component=vacuum_component)
    n_cross_B_smooth, gamma_surf_smooth = n_cross_B, gamma_surf
    
    dtheta = 2 * torch.pi / ntheta
    dphi_smooth = 2 * torch.pi / nphi_smooth

    def B_ext_of_phi(ii_pt):

        # window function
        center = phi_target[ii_pt]

        """ smooth part of the integral """

        # biot-savart kernel
        rprime = X_target[ii_pt] - gamma_surf_smooth # (nphi, ntheta, 3)
        norm_rprime_cubed = (torch.sqrt(torch.sum(rprime**2, dim=-1, keepdims=True))**3) # (nphi, ntheta, 1)
        kernel = rprime / norm_rprime_cubed
        # cross product
        integrand_smooth = torch.linalg.cross(kernel, n_cross_B_smooth, dim=-1) # (nphi, ntheta, 3)
        
        w_smooth = periodic_window_func(phi_smooth, center, sharpness, half_width) # test window function
        integral_smooth =  (1.0 / (4 * torch.pi) ) * torch.sum((integrand_smooth * (1-w_smooth)[:,None,None]) * dtheta * dphi_smooth, dim=(0,1)) # (3,)

        """ near-singular part of the integral """
        use_trap = False

        if use_trap:
            # TODO: remove this `if`; just here for testing.
            # interpolate
            nphi_sing = 8192
            phi_sing = torch.linspace(0, 2*torch.pi, nphi_sing+1)[:-1] # phi values of quad points
            n_cross_B_sing = fourier_interp1d(n_cross_B, phi_sing, period = period, dim=0) # (nphi, ntheta, 3)
            gamma_surf_sing = fourier_interp1d(gamma_surf, phi_sing, period = period, dim=0) # (nphi, ntheta, 3)
            dphi_sing = 2 * torch.pi / nphi_sing

            # biot-savart kernel
            rprime = X_target[ii_pt] - gamma_surf_sing # (nphi, ntheta, 3)
            norm_rprime_cubed = (torch.sqrt(torch.sum(rprime**2, dim=-1, keepdims=True))**3) # (nphi, ntheta, 1)
            kernel = rprime / norm_rprime_cubed
            # cross product
            integrand_sing = torch.linalg.cross(kernel, n_cross_B_sing, dim=-1) # (nphi, ntheta, 3)

            w_sing = periodic_window_func(phi_sing, center, sharpness, half_width) # test window function
            integral_sing = (1.0 / (4 * torch.pi) ) * torch.sum((integrand_sing * w_sing[:,None,None]) * dtheta * dphi_sing, dim=(0,1)) # (3,)
        else:
            # break the window into intervals
            # [a,b] should be wide enough such that window is ~0 outside
            a = center - 3.0 * half_width
            b = center + 3.0 * half_width
            dw = (b - a) / n_intervals
            np.testing.assert_allclose(periodic_window_func(a, center, sharpness, half_width), 0.0, atol=1e-7)
            np.testing.assert_allclose(periodic_window_func(b, center, sharpness, half_width), 0.0, atol=1e-7)

            integral_sing = torch.zeros((3,))
            for ii_int in range(n_intervals):
                lb = (a + ii_int * dw) 
                ub = (a + (ii_int + 1) * dw) 
                nodes, weights = gauss_quadrature_nodes_weights(lb, ub)

                # make nodes periodic
                nodes = nodes % period

                # interpolate n_cross_B and gamma_surf at the nodes
                n_cross_B_sing = fourier_interp1d(n_cross_B, nodes, period = period, dim=0) # (nphi, ntheta, 3)
                gamma_surf_sing = fourier_interp1d(gamma_surf, nodes, period = period, dim=0) # (nphi, ntheta, 3)

                # biot-savart kernel
                rprime = X_target[ii_pt] - gamma_surf_sing # (nphi, ntheta, 3)
                norm_rprime_cubed = (torch.sqrt(torch.sum(rprime**2, dim=-1, keepdims=True))**3) # (nphi, ntheta, 1)
                kernel = rprime / norm_rprime_cubed
                integrand_sing = torch.linalg.cross(kernel, n_cross_B_sing, dim=-1) # (nphi, ntheta, 3)

                # integrate in theta
                w_sing = periodic_window_func(nodes, center, sharpness, half_width) # (nphi,)
                dint_sing = (1.0 / (4 * torch.pi) ) * torch.sum((integrand_sing * w_sing[:,None,None]) * dtheta * weights[:,None,None], dim=(0,1)) # (3,)
                integral_sing += dint_sing

        integral = integral_smooth + integral_sing


        return integral
    
    B_ext = torch.stack([B_ext_of_phi(ii) for ii in range(n_target)]).T

    return B_ext

@lru_cache(maxsize=8)
def grad_B_external_on_axis_taylor(self, r=0.1, ntheta=256, nphi=1024, X_target=[], vacuum_component=False):
    """Compute grad_B_external on the magnetic axis using the virtual casing principle and 
    the Taylor expansion of B,
        grad_Bext(r) = (1/4pi) int [ grad_k(r,r') x (n(r') x B(r')) ] dtheta dphi
    If vacuum_component is True, then the integrand is computed assuming p2=I2=0.
        
    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
        X_target (tensor, optional): (n, 3) tensor of n target points inside the surface
            of radius r at which to evaluate B_external. The points do not necessarily need
            to be on magnetic axis. Defaults to (nphi, 3) tensor of points on the magnetic 
            axis, uniformly spaced in the axis cylindrical phi.
        vacuum_component (bool, optional): If true, computes the integral assuming (p2=I2=0).
            Defaults to False.
    Returns:
        (tensor): (3, 3, n) tensor of evaluations of B_external. 
            The gradient is a symmetric matrix at each target point.
    """
    # TODO: rename this method since target points need not be on axis

    if len(X_target) == 0:
        X_target = torch.clone(self.XYZ0.T) # (nphi, 3)
    n_target = len(X_target)

    # interpolate
    n_cross_B_interp, gamma_surf_interp = build_virtual_casing_interpolants(self, r=r, ntheta=ntheta, nphi=nphi, vacuum_component=vacuum_component)
    
    # compute integral
    B_ext = grad_B_external_integral(X_target, gamma_surf_interp, n_cross_B_interp, ntheta, nphi)

    return B_ext

@torch.compile
def grad_B_external_integral(X_target, gamma_surf_interp, n_cross_B_interp, ntheta, nphi):
    """Evaluate the integral for grad_B_external_on_axis_taylor. This function has been separated
    to enable torch.compile. Compilation speeds up repeated integral evaluation, but may slow the first 
    evaluation.

    Args:
        X_target (tensor): (n, 3) tensor of n target points inside the surface
            of radius r at which to evaluate B_external. The points do not necessarily need
            to be on magnetic axis.
        gamma_surf_interp (tensor): (nphi, ntheta, 3) tensor of flux surface coordinates.
        n_cross_B_interp (tensor): (nphi, ntheta, 3) tensor of n x B on the flux surface.
        ntheta (int): number of theta quadrature points.
        nphi (int): number of phi quadrature points.

    Returns:
        B_ext: (tensor): (3, 3, n) tensor of evaluations of B_external.
    """
    n_target = len(X_target)
    dtheta = 2 * torch.pi / ntheta
    dphi = 2 * torch.pi / nphi
    eye = torch.eye(3)
    B_ext = torch.zeros((3, 3, n_target))
    for ii in range(n_target):
        # biot-savart kernel
        rprime = X_target[ii] - gamma_surf_interp # (nphi, ntheta, 3)
        norm_rprime_cubed = (torch.sqrt(torch.sum(rprime**2, dim=-1, keepdims=True))**3) # (nphi, ntheta, 1)
        norm_rprime_fifth = (torch.sqrt(torch.sum(rprime**2, dim=-1, keepdims=True))**5) # (nphi, ntheta, 1)
        second_term = 3 * rprime / norm_rprime_fifth

        for jj in range(3):

            dkernel_by_djj = eye[jj].reshape((1,1,-1))/norm_rprime_cubed - rprime[:,:,jj][:,:,None] * second_term

            # cross product
            integrand = torch.linalg.cross(dkernel_by_djj, n_cross_B_interp, dim=-1) # (nphi, ntheta, 3)

            B_ext[:, jj, ii] =  (1.0 / (4 * torch.pi) ) * torch.sum(integrand *  dtheta * dphi, dim=(0,1)) # (3,)
    return B_ext

@lru_cache(maxsize=8)
def B_external_on_axis_corrected(self, r=0.1, ntheta=256, nphi=1024):
    """Compute B_external on the magnetic axis using the virtual casing principle. This method
    corrects the near axis virtual casing computation using a shift. It is exact for
    vacuum configurations.

    The solution is expressed as
        B_ext(r) = B_vac(r) + int k(r,r') x (n(r') x B(r')) dtheta dphi - int k_vac(r,r') x (n_vac(r') x B_vac(r')) dtheta dphi
    where k is the biot-savart kernel and n is the surface normal.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.

    Returns:
        (tensor): (3, nphi) tensor of evaluations of B_external on the magnetic axis nodes.
    """
    Bvac = self.Bfield_cartesian() # (3, nphi)
    Bext = B_external_on_axis_taylor(self, r=r, ntheta=ntheta, nphi=nphi) # (3, nphi)
    Bext_vac = B_external_on_axis_taylor(self, r=r, ntheta=ntheta, nphi=nphi, vacuum_component=True) # (3, nphi)
    B_ext = Bvac + (Bext - Bext_vac)

    return B_ext

@lru_cache(maxsize=8)
def grad_B_external_on_axis_corrected(self, r=0.1, ntheta=256, nphi=1024):
    """Compute grad_B_external on the magnetic axis using the virtual casing principle.
    This method corrects the near axis virtual casing computation using a shift. It is exact for
    vacuum configurations.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
    Returns:
        (tensor): (3, 3, n) tensor of evaluations of B_external. 
            The gradient is a symmetric matrix at each target point.
    """
    # compute Bext using Taylor expansion
    gradB_ext = B_external_on_axis_taylor(self, r=r, ntheta=ntheta, nphi=nphi) # (3, nphi)
    # compute correction terms
    grad_B_vac = self.grad_B_tensor_cartesian(vacuum_component=True) # (3, 3, ntarget)
    gradB_ext_vac = B_external_on_axis_taylor(self, r=r, ntheta=ntheta, nphi=nphi, vacuum_component=True) # (3, nphi)
    gradB_ext_corrected = grad_B_vac + (gradB_ext - gradB_ext_vac)
    return gradB_ext_corrected

@lru_cache(maxsize=8)
def build_virtual_casing_interpolants(self, r=0.1, ntheta=256, nphi=1024, vacuum_component=False):
    """Prepare interpolants for computing the virtual casing integral. In particular, interpolate
        n x B and the surface coordinates on a regular grid in (phi, theta).

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
        vacuum_component (bool, optional): If true, computes the interpolants assuming (p2=I2=0).
            Defaults to False.
    Returns:
        (tensor): (nphi, ntheta, 3) tensor of evaluations of the surface current n x B on the surface.
        (tensor): (nphi, ntheta, 3) tensor of evaluations of the Cartesian surface coordinates.
    """
    # components of integrand
    dvarphi_by_dphi = torch.clone(self.d_varphi_d_phi)
    n = self.surface_normal(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
    g = self.surface(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
    b = self.B_taylor(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
    nb = torch.linalg.cross(n, b) * dvarphi_by_dphi.reshape((-1,1,1))

    # map out full torus
    gamma_surf = torch.zeros((int(self.nfp * self.nphi), ntheta, 3))
    n_cross_B = torch.zeros((int(self.nfp * self.nphi), ntheta, 3))
    gamma_surf[ : self.nphi] = g
    n_cross_B[ : self.nphi] = nb
    for ii in range(1, self.nfp):
        g = rotate_nfp(g, 1, self.nfp)
        gamma_surf[ii * self.nphi : (ii+1) * self.nphi] = g
        nb = rotate_nfp(nb, 1, self.nfp)
        n_cross_B[ii * self.nphi : (ii+1) * self.nphi] = nb

    # interpolate only in phi
    period = 2 * torch.pi
    points = torch.linspace(0, period, nphi+1)[:-1] # linspace(..., endpoint=False)
    n_cross_B_interp = fourier_interp1d(n_cross_B, points, period = period, dim=0) # (nphi, ntheta, 3)
    gamma_surf_interp = fourier_interp1d(gamma_surf, points, period = period, dim=0) # (nphi, ntheta, 3)

    return n_cross_B_interp, gamma_surf_interp

@lru_cache(maxsize=8)
def build_virtual_casing_grid(self, r=0.1, ntheta=256, vacuum_component=False):
    """Evaluate the elements of the integrand for computing the virtual casing integral,
    n x B and gamma, on a grid in (phi, theta). No interplation is performed, the values
    are evaluated at the original nphi quadrature points.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        vacuum_component (bool, optional): If true, computes the interpolants assuming (p2=I2=0).
            Defaults to False.
    Returns:
        (tensor): (nphi, ntheta, 3) tensor of evaluations of the surface current n x B on the surface.
        (tensor): (nphi, ntheta, 3) tensor of evaluations of the Cartesian surface coordinates.
    """
    # components of integrand
    dvarphi_by_dphi = torch.clone(self.d_varphi_d_phi)
    n = self.surface_normal(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
    g = self.surface(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
    b = self.B_taylor(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
    nb = torch.linalg.cross(n, b) * dvarphi_by_dphi.reshape((-1,1,1))

    # map out full torus
    gamma_surf = torch.zeros((int(self.nfp * self.nphi), ntheta, 3))
    n_cross_B = torch.zeros((int(self.nfp * self.nphi), ntheta, 3))
    gamma_surf[ : self.nphi] = g
    n_cross_B[ : self.nphi] = nb
    for ii in range(1, self.nfp):
        g = rotate_nfp(g, 1, self.nfp)
        gamma_surf[ii * self.nphi : (ii+1) * self.nphi] = g
        nb = rotate_nfp(nb, 1, self.nfp)
        n_cross_B[ii * self.nphi : (ii+1) * self.nphi] = nb

    return n_cross_B, gamma_surf

def gauss_quadrature_nodes_weights(a,b):
    """Return the nodes and weights for 5th order Gauss-Legendre quadrature on [a,b].
    The quadrature, over [a,b], can be executed as
        integral = sum_i weights[i] * f(nodes[i]).

    Args:
        a (float): lower bound of integration
        b (float): upper bound of integration
        
    Returns:
        (nodes, weights): tuple of torch tensors each of length 5.
    """
    # weihts on [-1,1]
    w1 = 128 / 225
    w2 = (322 + 13 * torch.sqrt(torch.tensor(70.0))) / 900
    w3 = (322 - 13 * torch.sqrt(torch.tensor(70.0))) / 900
    weights = torch.tensor([w1, w2, w2, w3, w3])
    # nodes on [-1,1]
    p1 = 0.0
    p2 = (1 / 3) * torch.sqrt(5 - 2*torch.sqrt(torch.tensor(10/7)))
    p3 = (1 / 3) * torch.sqrt(5 + 2*torch.sqrt(torch.tensor(10/7)))
    nodes = torch.tensor([p1, p2, -p2, p3, -p3])
    # map to [a,b]
    weights = weights * (b - a) / 2
    nodes = nodes * (b - a) / 2 + (a + b) / 2
    return nodes, weights