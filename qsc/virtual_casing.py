#!/usr/bin/env python3

"""
Methods for computing the virtual casing integral.
"""

import logging
import numpy as np
from functools import lru_cache
import torch
from .util import rotate_nfp
from .fourier_tools import fourier_interp2d_regular_grid


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
        gamma_surf = self.surface(r, ntheta=ntheta) # (nphi, ntheta, 3)
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
        gamma_surf = self.surface(r, ntheta=ntheta) # (nphi, ntheta, 3)
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
            field. Defaults to False.

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


def B_external_on_axis_taylor(self, r=0.1, ntheta=256, nphi=1024, ntheta_eval=32, X_target=[]):
    """Compute B_external on the magnetic axis using the virtual casing principle.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
        ntheta_eval (int, optional): number of theta points at which to evaluate integrand prior
            to building interpolants.
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

    # components of integrand
    dvarphi_by_dphi = self.d_varphi_d_phi
    n = self.surface_normal(r=r, ntheta=ntheta_eval) # (nphi, ntheta, 3)
    g = self.surface(r=r, ntheta=ntheta_eval) # (nphi, ntheta, 3)
    b = self.B_taylor(r=r, ntheta=ntheta_eval) # (nphi, ntheta, 3)
    nb = torch.linalg.cross(n, b) * dvarphi_by_dphi.reshape((-1,1,1))

    # map out full torus
    gamma_surf = torch.zeros((int(self.nfp * self.nphi), ntheta_eval, 3))
    n_cross_B = torch.zeros((int(self.nfp * self.nphi), ntheta_eval, 3))
    for ii in range(self.nfp):
        g = rotate_nfp(g, 1, self.nfp)
        gamma_surf[ii * self.nphi : (ii+1) * self.nphi] = g
        nb = rotate_nfp(nb, 1, self.nfp)
        n_cross_B[ii * self.nphi : (ii+1) * self.nphi] = nb

    # interpolate
    n_cross_B_interp = fourier_interp2d_regular_grid(n_cross_B, nphi, ntheta) # (nphi, ntheta, 3)
    gamma_surf_interp = fourier_interp2d_regular_grid(gamma_surf, nphi, ntheta) # (nphi, ntheta, 3)

    dtheta = 2 * torch.pi / ntheta
    dphi = 2 * torch.pi / nphi

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

def B_external_on_axis_split(self, r=0.1, ntheta=256, nphi=1024, ntheta_eval=32, ntarget=0):
    """Compute B_external on the magnetic axis using the virtual casing principle. In this method
    we use the splitting method to compute the virtual casing to higher accuracy.

    The solution is expressed as
        B_ext(r) = B_vac(r) + int k(r,r') x (n(r') x B_nonvac(r')) dtheta dphi
    where k is the biot-savart kernel and n is the surface normal.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
        ntheta_eval (int, optional): number of theta points at which to evaluate integrand prior
            to building interpolants.
        ntarget (int, optional): number of target points to evaluate B_external on the magnetic axis.
            If ntarget is 0, it will default to nphi points on the magnetic axis, 
            uniformly spaced in the axis cylindrical phi.
    Returns:
        (tensor): (3, n) tensor of evaluations of B_external.
    """

    if ntarget == 0:
        ntarget = self.nphi
    X_target, _, idx = self.subsample_axis_nodes(ntarget) # (3, ntarget), (ntarget,)
    X_target = X_target.T

    # B_vac at axis nodes
    B_vac = self.Bfield_cartesian()[:,idx] # (3, ntarget)

    if (self.p2 == 0.0) and (self.I2 == 0.0):
        # in vacuum, return vacuum solution
        return B_vac

    # get interpolated data
    n_cross_B_interp, gamma_surf_interp = build_virtual_casing_interpolants_split(self, r=r, ntheta=ntheta, nphi=nphi, ntheta_eval=ntheta_eval)

    dtheta = 2 * torch.pi / ntheta
    dphi = 2 * torch.pi / nphi

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

    # nonvacuum component
    B_ext_nonvac = torch.stack([B_ext_of_phi(ii) for ii in range(ntarget)]).T # (3, ntarget)

    B_ext = B_vac + B_ext_nonvac # (3, ntarget)

    return B_ext

def grad_B_external_on_axis_split(self, r=0.1, ntheta=256, nphi=1024, ntheta_eval=32, ntarget=0):
    """Compute grad_B_external on the magnetic axis using the virtual casing principle.
    This function splits the magnetic field into vacuum and non-vacuum components to ensure
    accuracy of the computation in vacuum.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
        ntheta_eval (int, optional): number of theta points at which to evaluate integrand prior
            to building interpolants.
        ntarget (int, optional): number of target points to evaluate B_external on the magnetic axis.
            If ntarget is 0, it will default to nphi points on the magnetic axis, 
            uniformly spaced in the axis cylindrical phi.
    Returns:
        (tensor): (3, 3, n) tensor of evaluations of B_external. 
            The gradient is a symmetric matrix at each target point.
    """
    if ntarget == 0:
        ntarget = self.nphi
    X_target, _, idx = self.subsample_axis_nodes(ntarget) # (3, ntarget), (ntarget,)
    X_target = X_target.T

    # grad_B_vac at axis nodes
    grad_B_vac = self.grad_B_tensor_cartesian(vacuum_component=True)[:,:,idx] # (3, 3, ntarget)

    if (self.p2 == 0.0) and (self.I2 == 0.0):
        # in vacuum, return vacuum solution
        return grad_B_vac

    # get interpolated data
    surface_current_interp, gamma_surf_interp = build_virtual_casing_interpolants_split(self, r=r, ntheta=ntheta, nphi=nphi, ntheta_eval=ntheta_eval)

    dtheta = 2 * torch.pi / ntheta
    dphi = 2 * torch.pi / nphi
    eye = torch.eye(3)
    
    # compute the non-vacuum component with the integral
    grad_B_ext_nonvac = torch.zeros((3, 3, ntarget))
    for ii in range(ntarget):
        # biot-savart kernel
        rprime = X_target[ii] - gamma_surf_interp # (nphi, ntheta, 3)
        norm_rprime_cubed = (torch.sqrt(torch.sum(rprime**2, dim=-1, keepdims=True))**3) # (nphi, ntheta, 1)
        norm_rprime_fifth = (torch.sqrt(torch.sum(rprime**2, dim=-1, keepdims=True))**5) # (nphi, ntheta, 1)
        second_term = 3 * rprime / norm_rprime_fifth

        for jj in range(3):

            dkernel_by_djj = eye[jj].reshape((1,1,-1))/norm_rprime_cubed - rprime[:,:,jj][:,:,None] * second_term

            # cross product
            integrand = torch.linalg.cross(dkernel_by_djj, surface_current_interp, dim=-1) # (nphi, ntheta, 3)

            grad_B_ext_nonvac[:, jj, ii] =  (1.0 / (4 * torch.pi) ) * torch.sum(integrand *  dtheta * dphi, dim=(0,1)) # (3,)
    
    grad_B_ext = grad_B_vac + grad_B_ext_nonvac # (3, 3, ntarget)

    return grad_B_ext


@lru_cache(maxsize=32)
def B_external_on_axis_nodes(self, r=0.1, ntheta=256, nphi=1024, ntheta_eval=32, ntarget=0):
    """Compute B_external on the magnetic axis using the virtual casing principle at
    the quadrature points on the magnetic axis.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
        ntheta_eval (int, optional): number of theta points at which to evaluate integrand prior
            to building interpolants. 
        ntarget (int, optional): number of target points to evaluate B_external on the magnetic axis.
            If ntarget is 0, it will default to nphi points on the magnetic axis, 
            uniformly spaced in the axis cylindrical phi.
    Returns:
        (tensor): (3, ntarget) tensor of evaluations of B_external.
    """
    if ntarget == 0:
        ntarget = self.nphi
    Xtarget = self.subsample_axis_nodes(ntarget)[0] # (3, ntarget)
    # return B_external_on_axis(self, r=r, ntheta=ntheta, nphi=nphi, ntheta_eval=ntheta_eval, X_target=Xtarget.T)
    return B_external_on_axis_split(self, r=r, ntheta=ntheta, nphi=nphi, ntheta_eval=ntheta_eval, ntarget=ntarget)

def B_external_on_axis(self, r=0.1, ntheta=256, nphi=1024, ntheta_eval=32, X_target=[]):
    """Compute B_external on the magnetic axis using the virtual casing principle.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
        ntheta_eval (int, optional): number of theta points at which to evaluate integrand prior
            to building interpolants. 
        X_target (tensor, optional): (n, 3) tensor of n target points inside the surface
            of radius r at which to evaluate B_external. The points do not necessarily need
            to be on magnetic axis. Defaults to (nphi, 3) tensor of points on the magnetic 
            axis, uniformly spaced in the axis cylindrical phi.
    Returns:
        (tensor): (3, n) tensor of evaluations of B_external.
    """
    if len(X_target) == 0:
        X_target = self.XYZ0.T # (ntarget, 3)
    n_target = len(X_target)

    # get interpolated data
    surface_current, gamma_surf_interp = build_virtual_casing_interpolants(self, r=r, ntheta=ntheta, nphi=nphi, ntheta_eval=ntheta_eval)

    dtheta = 2 * torch.pi / ntheta
    dphi = 2 * torch.pi / nphi

    def B_ext_of_phi(ii):
        """ Compute B_external by integrating over the entire device. """

        # biot-savart kernel
        rprime = X_target[ii] - gamma_surf_interp # (nphi, ntheta, 3)
        norm_rprime_cubed = (torch.sqrt(torch.sum(rprime**2, dim=-1, keepdims=True))**3) # (nphi, ntheta, 1)
        kernel = rprime / norm_rprime_cubed

        # cross product
        integrand = torch.linalg.cross(kernel, surface_current, dim=-1) # (nphi, ntheta, 3)

        integral =  (1.0 / (4 * torch.pi) ) * torch.sum(integrand *  dtheta * dphi, dim=(0,1)) # (3,)

        return integral
    
    B_ext = torch.stack([B_ext_of_phi(ii) for ii in range(n_target)]).T
    return B_ext

@lru_cache(maxsize=32)
def grad_B_external_on_axis_nodes(self, r=0.1, ntheta=256, nphi=1024, ntheta_eval=32, ntarget=0):
    """Compute grad_B_external on the magnetic axis using the virtual casing principle at
    the quadrature points on the magnetic axis.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
        ntheta_eval (int, optional): number of theta points at which to evaluate integrand prior
            to building interpolants. 
        ntarget (int, optional): number of target points to evaluate B_external on the magnetic axis.
            If ntarget is 0, it will default to nphi points on the magnetic axis, 
            uniformly spaced in the axis cylindrical phi.
    Returns:
        (tensor): (3, 3, ntarget) tensor of evaluations of B_external. 
            The gradient is a symmetric matrix at each target point.
    """
    if ntarget == 0:
        ntarget = self.nphi
    Xtarget = self.subsample_axis_nodes(ntarget)[0] # (3, ntarget)

    # return grad_B_external_on_axis(self, r=r, ntheta=ntheta, nphi=nphi, ntheta_eval=ntheta_eval, X_target=Xtarget.T)
    return grad_B_external_on_axis_split(self, r=r, ntheta=ntheta, nphi=nphi, ntheta_eval=ntheta_eval, ntarget=ntarget)

def grad_B_external_on_axis(self, r=0.1, ntheta=256, nphi=1024, ntheta_eval=32, X_target=[]):
    """Compute grad_B_external on the magnetic axis using the virtual casing principle.

    This function calculates the gradient of the external magnetic field by evaluating
    a surface integral over a flux surface. The integral uses the virtual casing principle
    where the surface current is represented as:
        j(x') = I * (partial r / partial varphi) - G * (partial r / partial theta)
    
    The gradient of B_external at a point x is computed through numerical integration of:
        grad B_external(x) = (1/4π) ∮ (grad k) x j dS'

    over a flux surface. grad k is the gradient of the Biot-Savart kernel, k(r,r') = (r-r')/|r-r'|^3,
        grad k(r,r') = I / |r-r'|^3 - 3 (r - r') (r-r')^T / |r-r'|^5.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
        ntheta_eval (int, optional): number of theta points at which to evaluate integrand prior
            to building interpolants.
        X_target (tensor, optional): (n, 3) tensor of n target points inside the surface
            of radius r at which to evaluate B_external. The points do not necessarily need
            to be on magnetic axis. Defaults to (nphi, 3) tensor of points on the magnetic 
            axis, uniformly spaced in the axis cylindrical phi.
    Returns:
        (tensor): (3, 3, n) tensor of evaluations of B_external. 
            The gradient is a symmetric matrix at each target point.
    """
    if len(X_target) == 0:
        X_target = self.XYZ0.T # (nphi, 3)
    n_target = len(X_target)

    # get interpolated data
    surface_current, gamma_surf_interp = build_virtual_casing_interpolants(self, r=r, ntheta=ntheta, nphi=nphi, ntheta_eval=ntheta_eval)

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
            integrand = torch.linalg.cross(dkernel_by_djj, surface_current, dim=-1) # (nphi, ntheta, 3)

            B_ext[:, jj, ii] =  (1.0 / (4 * torch.pi) ) * torch.sum(integrand *  dtheta * dphi, dim=(0,1)) # (3,)

    return B_ext

@lru_cache(maxsize=32)
def build_virtual_casing_interpolants(self, r=0.1, ntheta=256, nphi=1024, ntheta_eval=32):
    """Interpolate the surface current n x B and the surface coordinates on the flux surface.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
        ntheta_eval (int, optional): number of theta points at which to evaluate integrand prior
            to building interpolants.
        X_target (tensor, optional): (n, 3) tensor of n target points inside the surface
            of radius r at which to evaluate B_external. The points do not necessarily need
            to be on magnetic axis. Defaults to (nphi, 3) tensor of points on the magnetic 
            axis, uniformly spaced in the axis cylindrical phi.
    Returns:
        (tensor): (nphi, ntheta, 3) tensor of evaluations of the surface current n x B on the surface.
        (tensor): (nphi, ntheta, 3) tensor of evaluations of the Cartesian surface coordinates.

    """
    I = 0.0
    G = self.G0
    if self.order != 'r1':
        I += r**2 * self.I2
        G += r**2 * self.G2

    # components of integrand
    dvarphi_by_dphi = self.d_varphi_d_phi
    dr_by_dvarphi = self.dsurface_by_dvarphi(r=r, ntheta=ntheta_eval) # (nphi, ntheta, 3)
    dr_by_dtheta = self.dsurface_by_dtheta(r=r, ntheta=ntheta_eval) # (nphi, ntheta, 3)
    d = (I * dr_by_dvarphi - G * dr_by_dtheta) * dvarphi_by_dphi.reshape((-1,1,1))
    g = self.surface(r=r, ntheta=ntheta_eval) # (nphi, ntheta, 3)

    # get surface and tangents across all nfp
    gamma_surf = torch.zeros((int(self.nfp * self.nphi), ntheta_eval, 3))
    diff = torch.zeros((int(self.nfp * self.nphi), ntheta_eval, 3))
    for ii in range(self.nfp):
        g = rotate_nfp(g, 1, self.nfp)
        gamma_surf[ii * self.nphi : (ii+1) * self.nphi] = g
        d = rotate_nfp(d, 1, self.nfp)
        diff[ii * self.nphi : (ii+1) * self.nphi] = d

    # interpolate
    surface_current = fourier_interp2d_regular_grid(diff, nphi, ntheta) # (nphi, ntheta, 3)
    gamma_surf_interp = fourier_interp2d_regular_grid(gamma_surf, nphi, ntheta) # (nphi, ntheta, 3)

    return surface_current, gamma_surf_interp

@lru_cache(maxsize=32)
def build_virtual_casing_interpolants_split(self, r=0.1, ntheta=256, nphi=1024, ntheta_eval=32):
    """Interpolate the nonvacuum surface current n x B_nonvac and the surface coordinates on the flux surface
    for the splitting method of computing the virtual casing integral.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 256.
        nphi (int, optional): number of phi quadrature points. Defaults to 1024.
        ntheta_eval (int, optional): number of theta points at which to evaluate integrand prior
            to building interpolants.
        X_target (tensor, optional): (n, 3) tensor of n target points inside the surface
            of radius r at which to evaluate B_external. The points do not necessarily need
            to be on magnetic axis. Defaults to (nphi, 3) tensor of points on the magnetic 
            axis, uniformly spaced in the axis cylindrical phi.
    Returns:
        (tensor): (nphi, ntheta, 3) tensor of evaluations of the surface current n x B_nonvac on the surface.
        (tensor): (nphi, ntheta, 3) tensor of evaluations of the Cartesian surface coordinates.

    """
    # components of integrand
    dvarphi_by_dphi = self.d_varphi_d_phi
    n = self.surface_normal(r=r, ntheta=ntheta_eval) # (nphi, ntheta, 3)
    g = self.surface(r=r, ntheta=ntheta_eval) # (nphi, ntheta, 3)
    b = self.B_taylor(r=r, ntheta=ntheta_eval) # (nphi, ntheta, 3)
    b_vac = self.B_taylor(r=r, ntheta=ntheta_eval, vacuum_component=True) # (nphi, ntheta, 3)
    b_nonvac = b - b_vac # (nphi, ntheta, 3)
    nb = torch.linalg.cross(n, b_nonvac) * dvarphi_by_dphi.reshape((-1,1,1))

    # map out full torus
    gamma_surf = torch.zeros((int(self.nfp * self.nphi), ntheta_eval, 3))
    n_cross_B_nonvac = torch.zeros((int(self.nfp * self.nphi), ntheta_eval, 3))
    for ii in range(self.nfp):
        g = rotate_nfp(g, 1, self.nfp)
        gamma_surf[ii * self.nphi : (ii+1) * self.nphi] = g
        nb = rotate_nfp(nb, 1, self.nfp)
        n_cross_B_nonvac[ii * self.nphi : (ii+1) * self.nphi] = nb

    # interpolate
    surface_current_interp = fourier_interp2d_regular_grid(n_cross_B_nonvac, nphi, ntheta) # (nphi, ntheta, 3)
    gamma_surf_interp = fourier_interp2d_regular_grid(gamma_surf, nphi, ntheta) # (nphi, ntheta, 3)

    return surface_current_interp, gamma_surf_interp
