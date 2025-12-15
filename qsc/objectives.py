#!/usr/bin/env python3

"""
Metrics for optimization.
"""

import logging
import numpy as np
import torch

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def Bfield_axis_mse(self, B_target):
    '''
    Integrated mean-squared error between the Cartesian magnetic field on axis
    and a target magnetic field over the magnetic axis,
            Loss = (1/2) int |B - B_target|**2 dl

    Args:
        B_target: (3, nphi) tensor of target magnetic field values.
    '''
    B = self.Bfield_cartesian() # (3, nphi)
    dphi = torch.diff(self.phi)[0]
    d_l_d_phi = self.d_l_d_phi
    dl = d_l_d_phi * dphi # (nphi,)
    loss = 0.5 * torch.sum(torch.sum((B - B_target)**2, dim=0) * dl) # scalar tensor
    return loss

def grad_B_tensor_cartesian_mse(self, gradB_target):
    '''
    Integrated mean-squared error between the gradient of the Cartesian magnetic field on axis
    and a target magnetic field over the magnetic axis,
            Loss = (1/2) int |gradB - gradB_target|**2 dl

    Args:
        gradB_target (tensor): (3, 3, nphi) tensor of target the gradient field values.

    Return:
        loss (tensor): float tensor of the objective value.
    '''
    gradB = self.grad_B_tensor_cartesian() # (3, 3, nphi)
    dphi = np.diff(self.phi)[0]
    d_l_d_phi = self.d_l_d_phi
    dl = d_l_d_phi * dphi # (nphi,)
    loss = 0.5 * torch.sum(torch.sum((gradB - gradB_target)**2, dim=(0,1)) * dl) # scalar tensor
    return loss

def grad_grad_B_tensor_cartesian_mse(self, grad_grad_B_target):
    '''
    Integrated mean-squared error between the second derivative tensor
    of the Cartesian magnetic field on axis and a target magnetic field over the magnetic axis,
            Loss = (1/2) int |grad_grad_B - grad_gradB_target|**2 dl

    Args:
        gradB_target (tensor): (3, 3, nphi) tensor of target the gradient field values.

    Return:
        loss (tensor): float tensor of the objective value.
    '''
    grad_grad_B = self.grad_grad_B_tensor_cartesian() # (3, 3, 3, nphi)
    dphi = np.diff(self.phi)[0]
    d_l_d_phi = self.d_l_d_phi
    dl = d_l_d_phi * dphi # (nphi,)
    loss = 0.5 * torch.sum(torch.sum((grad_grad_B - grad_grad_B_target)**2, dim=(0,1,2)) * dl) # scalar tensor
    return loss

def downsample_axis(self, nphi):
    """This convenience function computes the Cartesian axis coordinates and derivatives of the
    arc length at evenly spaced (in phi) points along the magnetic axis.

    Args:
        nphi (int): number of quadrature points

    Returns:
        (tensor): (3, nphi) tensor of points on the magnetic axis.
        (tensor): (nphi,) tensor of derivatives of the arclength at the quadrature points with
            respect to the cylindrical angle on axis, dl/dphi.
    """
    phi = torch.tensor(np.linspace(0, 2 * torch.pi / self.nfp, nphi, endpoint=False))
    R0 = torch.zeros(nphi)
    Z0 = torch.zeros(nphi)
    R0p = torch.zeros(nphi)
    Z0p = torch.zeros(nphi)
    for jn in range(0, self.nfourier):
        n = jn * self.nfp
        sinangle = torch.sin(n * phi)
        cosangle = torch.cos(n * phi)
        R0 += self.rc[jn] * cosangle + self.rs[jn] * sinangle
        Z0 += self.zc[jn] * cosangle + self.zs[jn] * sinangle
        R0p += self.rc[jn] * (-n * sinangle) + self.rs[jn] * (n * cosangle)
        Z0p += self.zc[jn] * (-n * sinangle) + self.zs[jn] * (n * cosangle)

    # Cartesian coords
    xyz = torch.stack((R0 * torch.cos(phi), R0 * torch.sin(phi), Z0)) # (3, nphi)
    d_l_d_phi = torch.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
    return xyz, d_l_d_phi

def subsample_axis_nodes(self, ntarget):
    """This convenience function computes the Cartesian axis coordinates and derivatives of the
    arc length at a subset of the quadrature points on the magnetic axis.

    Args:
        ntarget (int): number of quadrature points between 1 and nphi.

    Returns:
        (tensor): (3, ntarget) tensor of points on the magnetic axis.
        (tensor): (ntarget,) tensor of derivatives of the arclength at the quadrature points with
            respect to the cylindrical angle on axis, dl/dphi.
        (tensor): (ntarget,) tensor of integer indexes of the points on axis.
    """
    # index of points on axis
    idx = torch.tensor(np.linspace(0, self.nphi, ntarget, endpoint=False, dtype=int))
    xyz = self.XYZ0[:, idx] # (3, nphi) tensor of points on the magnetic axis
    d_l_d_phi = self.d_l_d_phi[idx] # (nphi,) tensor of derivatives of the arc length with respect to phi

    return xyz, d_l_d_phi, idx

def B_external_on_axis_mse(self, B_target, r, ntheta=256, nphi=1024):
    '''
    Integrated mean-squared error between the Cartesian external magnetic field on axis
    and a target magnetic field over the magnetic axis,
            Loss = (1/2) int |B - B_target|**2 dl
    B is computed by the virtual casing integral by integrating over a surface of
    radius r.

    Args:
        B_target (tensor): (3, n) tensor of target magnetic field values.
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points for virtual casing integral. Defaults to 256.
        nphi (int, optional): number of phi quadrature points for virtual casing integral. Defaults to 1024.

    Returns:
        (tensor): (1,) Loss value as a scalar tensor.
    '''
    Bext_vc = self.B_external_on_axis(r=r, ntheta=ntheta, nphi=nphi) # (3, n)
    # dl = self.d_l_d_phi * self.d_phi # (nphi,)
    dl = torch.clone(self.d_l)
    loss = 0.5 * torch.sum(torch.sum((Bext_vc - B_target)**2, dim=0) * dl) # scalar tensor
    return loss

def grad_B_external_on_axis_mse(self, grad_B_target, r, ntheta=256, nphi=1024):
    '''
    Integrated mean-squared error between the gradient of the Cartesian external magnetic field on axis
    and a target gradient field over the magnetic axis,
            Loss = (1/2) int |grad_B_external - grad_B_target|**2 dl
    grad_B_external is computed by the virtual casing integral by integrating over a surface of
    radius r.

    Args:
        grad_B_target (tensor): (3, 3, n) tensor of target magnetic field values.
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points for virtual casing integral. Defaults to 256.
        nphi (int, optional): number of phi quadrature points for virtual casing integral. Defaults to 1024.

    Returns:
        (tensor): (1,) Loss value as a scalar tensor.
    '''
    grad_Bext = self.grad_B_external_on_axis(r=r, ntheta=ntheta, nphi=nphi) # (3, 3, n)
    # dl = self.d_l_d_phi * self.d_phi # (nphi,)
    dl = torch.clone(self.d_l)
    loss = 0.5 * torch.sum(torch.sum((grad_Bext - grad_B_target)**2, dim=(0,1)) * dl) # scalar tensor
    return loss

def total_derivative(self, loss):
    """Get the total derivative of a loss function with respect to the DOFs.
    The derivative of loss(sigma, iota, sigma_vac, iota_vac) with respect to the DOFS, x, is
        dloss/dx = dloss/d(sigma,iota) * d(sigma,iota)/dx + dloss/dx
                + dloss/d(sigma_vac,iota_vac) * d(sigma_vac,iota_vac)/dx 
    Example:
        stel = Qsc.from_paper("precise QA", order='r1')
        gradB_target = 1.34 + torch.clone(stel.grad_B_tensor_cartesian()).detach()
        loss = stel.grad_B_tensor_cartesian_mse(gradB_target)
        dloss_by_dofs = stel.total_derivative(loss) # list

    Args:
        loss (tensor): evaluated loss function

    Returns:
        list: list of gradients of loss with respect to dofs. List order matches get_dofs() function.
    """

    # compute gradient 
    self.zero_grad()
    loss.backward(retain_graph=True)

    # get dofs
    dofs = self.get_dofs(as_tuple=True)

    # make sure derivatives are not None
    partialloss_by_partialsigma = self.sigma.grad
    partialloss_by_partialiota = self.iota.grad 
    partialloss_by_partialsigma_vac = self.sigma_vac.grad
    partialloss_by_partialiota_vac = self.iota_vac.grad 
    if partialloss_by_partialsigma is None:
        partialloss_by_partialsigma = torch.zeros(self.nphi)
    if partialloss_by_partialiota is None:
        partialloss_by_partialiota = torch.zeros(1)
    if partialloss_by_partialsigma_vac is None:
        partialloss_by_partialsigma_vac = torch.zeros(self.nphi)
    if partialloss_by_partialiota_vac is None:
        partialloss_by_partialiota_vac = torch.zeros(1)

    # solve adjoint
    partialloss_by_partialz = torch.clone(partialloss_by_partialsigma).detach()
    partialloss_by_partialz[0] = torch.clone(partialloss_by_partialiota).detach()
    partialloss_by_partial_dofs = self.dresidual_by_ddof_vjp(partialloss_by_partialz) # tuple; sorted by dof

    # solve adjoint for vacuum components
    partialloss_by_partialz_vac = torch.clone(partialloss_by_partialsigma_vac).detach()
    partialloss_by_partialz_vac[0] = torch.clone(partialloss_by_partialiota_vac).detach()
    partialloss_by_partial_dofs_vac = self.dresidual_vac_by_ddof_vjp(partialloss_by_partialz_vac) # tuple; sorted by dof

    # chain rule dloss/dz * dz/dx + dloss/dz_vac * dz_vac/dx + dloss/dx
    dloss_by_ddofs = []
    for ii, x in enumerate(dofs):
        dloss_by_dx = torch.zeros_like(x)
        if partialloss_by_partial_dofs[ii] is not None:
            dloss_by_dx += partialloss_by_partial_dofs[ii]
        if partialloss_by_partial_dofs_vac[ii] is not None:
            dloss_by_dx += partialloss_by_partial_dofs_vac[ii]
        if x.grad is not None:
            dloss_by_dx += x.grad

        dloss_by_ddofs.append(dloss_by_dx)

    return dloss_by_ddofs

def surface_integral(self, X, r, vacuum_component=False):
    """Take the surface integral of a function X(r, theta, phi),
        I = int X(r, theta, phi) * dA.
    The surface integral is taken over one field period using the Trapezoidal rule
    in both theta and phi (not varphi). The result is multiplied by nfp.

    Args:
        X (tensor): (nphi, ntheta) tensor of function values on the surface.
            It is assumed that the points are uniformly spaced in phi (not varphi)
            and theta.
        r (float): radius of flux surface
        vacuum_component (bool): if True, the area element of the vacuum surface (p2=I2=0)
            is used. Default False.

    Returns:
        (tensor): (1,) tensor of the integral value.
    """
    nphi, ntheta = X.shape
    assert nphi == self.nphi, f"X.shape[0] = {nphi} does not match nphi = {self.nphi}"
    dphi = 2 * torch.pi / self.nfp / nphi
    dtheta = 2 * torch.pi / ntheta
    dA = self.surface_area_element(r=r, ntheta=ntheta, vacuum_component=vacuum_component) *  self.d_varphi_d_phi[:, None] # (nphi, ntheta)
    integral = self.nfp * torch.sum(X * dA * dphi * dtheta)  # scalar tensor
    return integral