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
    Integrated mean-squared error between the Cartesian magnetic field on axis
    and a target magnetic field over the magnetic axis,
            Loss = (1/2) int |B - B_target|**2 dl

    Args:
        B_target: (3, nphi) tensor of target magnetic field values.
        r: the near-axis radius
        theta: the Boozer poloidal angle vartheta (= theta-N*phi)
    '''
    gradB = self.grad_B_tensor_cartesian() # (3, 3, nphi)
    dphi = np.diff(self.phi)[0]
    d_l_d_phi = self.d_l_d_phi
    dl = d_l_d_phi * dphi # (nphi,)
    loss = 0.5 * torch.sum(torch.sum((gradB - gradB_target)**2, dim=(0,1)) * dl) # scalar tensor
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

def B_external_on_axis_mse(self, B_target, r, ntheta=256, nphi=1024):
    '''
    Integrated mean-squared error between the Cartesian external magnetic field on axis
    and a target magnetic field over the magnetic axis,
            Loss = (1/2) int |B - B_target|**2 dl
    B is computed by the virtual casing integral by integrating over a surface of
    radius r.

    Args:
        B_target (tensor): (3, ntarget) tensor of target magnetic field values.
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points for virtual casing integral. Defaults to 256.
        nphi (int, optional): number of phi quadrature points for virtual casing integral. Defaults to 1024.
    Returns:
        (tensor): (1,) Loss value as a scalar tensor.
    '''
    ntarget = B_target.shape[1]
    X_target, d_l_d_phi = self.downsample_axis(ntarget)
    Bext_vc = self.B_external_on_axis(r=r, ntheta=ntheta, nphi=nphi, X_target = X_target.T) # (3, ntarget)
    dphi = (2 * torch.pi / self.nfp) / ntarget
    dl = d_l_d_phi * dphi # (ntarget,)
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
        grad_B_target (tensor): (3, 3, ntarget) tensor of target magnetic field values.
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points for virtual casing integral. Defaults to 256.
        nphi (int, optional): number of phi quadrature points for virtual casing integral. Defaults to 1024.
    Returns:
        (tensor): (1,) Loss value as a scalar tensor.
    '''
    ntarget = grad_B_target.shape[-1]
    X_target, d_l_d_phi = self.downsample_axis(ntarget)
    grad_Bext = self.grad_B_external_on_axis(r=r, ntheta=ntheta, nphi=nphi, X_target = X_target.T) # (3, 3, ntarget)
    dphi = (2 * torch.pi / self.nfp) / ntarget
    dl = d_l_d_phi * dphi # (ntarget,)
    loss = 0.5 * torch.sum(torch.sum((grad_Bext - grad_B_target)**2, dim=(0,1)) * dl) # scalar tensor
    return loss

def total_derivative(self, loss):
    """Get the total derivative of a loss function with respect to the DOFs.
        dloss/dx = dloss/d(sigma,iota) * d(sigma,iota)/dx + dloss/dx

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
    if partialloss_by_partialsigma is None:
        partialloss_by_partialsigma = torch.zeros(self.nphi)
    if partialloss_by_partialiota is None:
        partialloss_by_partialiota = torch.zeros(1)

    # solve adjoint
    partialloss_by_partialz = torch.clone(partialloss_by_partialsigma).detach()
    partialloss_by_partialz[0] = torch.clone(partialloss_by_partialiota).detach()
    partialloss_by_partial_dofs = self.dresidual_by_ddof_vjp(partialloss_by_partialz) # tuple; sorted by dof

    # chain rule dloss/dz * dz/dx + dloss/dx
    dloss_by_ddofs = []
    for ii, x in enumerate(dofs):
        dloss_by_dx = torch.zeros_like(x)
        if partialloss_by_partial_dofs[ii] is not None:
            dloss_by_dx += partialloss_by_partial_dofs[ii]
        if x.grad is not None:
            dloss_by_dx += x.grad

        dloss_by_ddofs.append(dloss_by_dx)

    return dloss_by_ddofs

