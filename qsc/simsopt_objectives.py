#!/usr/bin/env python3

"""
Metrics for optimization.
"""

import logging
import numpy as np
import torch
from .objectives import Bfield_axis_mse, grad_B_tensor_cartesian_mse, total_derivative
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from qsc import Qsc

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QscOptimizable(Qsc, Optimizable):
    def __init__(self, *args, **kwargs):
        Qsc.__init__(self, *args, **kwargs)
        Optimizable.__init__(self, x0=Qsc.get_dofs(self),
                             external_dof_setter=Qsc.set_dofs,
                             names=self.names)

class FieldError(Optimizable):
    def __init__(self, bs, qsc):
        """
        bs: biotsavart object
        qsc: an Optimizable Qsc object
        """
        self.bs = bs
        self.qsc = qsc
        Optimizable.__init__(self, depends_on=[bs, qsc])

    def field_error(self):
        """
        Sum-of-squares error in the field:
            (1/2) * int |B_axis - B_coil|^2 dl/dphi dphi
        where the integral is taken along the axis.
        """
        # evaluate coil field
        xyz = np.ascontiguousarray(self.qsc.XYZ0.detach().numpy().T) # (nphi, 3)
        xyz = np.ascontiguousarray(xyz)

        self.bs.set_points(xyz)
        B_coil = self.bs.B().T # (3, nphi)
        # compute loss
        loss = self.qsc.Bfield_axis_mse(torch.tensor(B_coil))
        return loss

    def dfield_error(self):
        """
        Derivative of the field error w.r.t all coil coeffs,
            axis coefs, and etabar.

        return: SIMSOPT Derivative object containing the derivatives
            of the field_error function with respect to the BiotSavart
            and Expansion DOFs.
        """
        # Qsc field
        B_qsc = self.qsc.Bfield_cartesian().T.detach().numpy() # (nphi, 3)

        # compute dl
        dphi = np.diff(self.qsc.phi)[0]
        d_l_d_phi = self.qsc.d_l_d_phi
        dl = (d_l_d_phi * dphi).detach().numpy().reshape((-1,1)) # (nphi, 1)

        # coil field
        xyz = np.ascontiguousarray(self.qsc.XYZ0.detach().numpy().T) # (nphi, 3)

        xyz = np.ascontiguousarray(xyz)
        self.bs.set_points(xyz)
        B_coil = self.bs.B() # (nphi, 3)

        # derivative with respect to biot savart dofs
        dJ_by_dbs = self.bs.B_vjp((B_coil - B_qsc)*dl) # Derivative object

        """ Derivative w.r.t. axis coeffs """
        # this part of the derivative treats B_coil as a constant, independent of the axis
        loss = self.field_error()
        dloss_by_ddofs = self.qsc.total_derivative(loss) # list

        # derivative of B_coil(xyz(axis_coeffs)) term
        # self.qsc.zero_grad()
        dB_by_dX_bs = self.bs.dB_by_dX() # (nphi, 3, 3)
        term21 = np.einsum("ji,jki->jk", ((B_coil - B_qsc) * dl), dB_by_dX_bs) # (nphi, 3)

        dofs = self.qsc.get_dofs(as_tuple=True)
        xyz = self.qsc.XYZ0.T # (nphi, 3)
        term2 = torch.autograd.grad(xyz, dofs, grad_outputs=torch.tensor(term21), retain_graph=True, allow_unused=True) # tuple

        derivs_axis = np.zeros(0)
        for ii, x in enumerate(dofs):
            # sum the two parts of the derivative
            dJ_by_dx = dloss_by_ddofs[ii].detach().numpy()
            if term2[ii] is not None:
                dJ_by_dx += term2[ii].detach().numpy()
            derivs_axis = np.append(derivs_axis, dJ_by_dx)

        # make a derivative object
        dJ_by_daxis = Derivative({self.qsc: derivs_axis})

        dJ = dJ_by_daxis + dJ_by_dbs

        return dJ