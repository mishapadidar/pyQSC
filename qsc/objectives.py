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