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
        r: the near-axis radius
        theta: the Boozer poloidal angle vartheta (= theta-N*phi)
    '''
    B = self.Bfield_cartesian() # (3, nphi)
    dphi = np.diff(self.phi)[0]
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