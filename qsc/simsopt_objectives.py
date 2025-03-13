#!/usr/bin/env python3

"""
Metrics for optimization.
"""

import logging
import numpy as np
import torch
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from qsc import Qsc

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QscOptimizable(Qsc, Optimizable):
    def __init__(self, *args, **kwargs):

        # for caching
        self.need_to_run_code = True

        Qsc.__init__(self, *args, **kwargs)
        Optimizable.__init__(self, x0=Qsc.get_dofs(self),
                             external_dof_setter=Qsc.set_dofs,
                             names=self.names)
        
    def recompute_bell(self, parent=None):
        """
        This function will get called any time any of the DOFs of the
        parent class change.
        """
        self.need_to_run_code = True
        return super().recompute_bell(parent)

    def calculate_or_cache(self):
        """Run the calculate() method, only if need_to_run_code is True.
        """
        if self.need_to_run_code:
            self.calculate()
            self.need_to_run_code = False


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
        self.qsc.calculate_or_cache()
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
            of the .J function with respect to the BiotSavart
            and Expansion DOFs.
        """
        self.qsc.calculate_or_cache()
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
    
    def J(self):
        """Compute the objective function, returning a float.

        Returns:
            float: objective function value.
        """
        return self.field_error().detach().numpy().item()
    
    @derivative_dec
    def dJ(self):
        """Compute the gradient of the objective function.

        Returns:
            array: gradient of the objective function as an np arrray.
        """
        return self.dfield_error()
    
class ExternalFieldError(Optimizable):
    def __init__(self, bs, qsc, r, ntheta=128, ntarget=32):
        """
        bs: biotsavart object
        qsc: an Optimizable Qsc object
        """
        self.bs = bs
        self.qsc = qsc
        self.r = r
        self.ntheta = ntheta
        self.ntarget = ntarget
        Optimizable.__init__(self, depends_on=[bs, qsc])

    def field_error(self):
        """
        Sum-of-squares error in the virtual-casing field:
            (1/2) * int |B_axis - B_coil|^2 dl/dphi dphi
        where the integral is taken along the axis.
        """
        self.qsc.calculate_or_cache()
        # evaluate coil field
        X_target, _ = self.qsc.downsample_axis(nphi=self.ntarget) # (3, nphi)
        X_target_np = X_target.detach().numpy().T # (nphi, 3)
        X_target_np = np.ascontiguousarray(X_target_np) # (nphi, 3)

        self.bs.set_points(X_target_np)
        B_coil = self.bs.B().T # (3, nphi)
        
        # compute loss
        loss = self.qsc.B_external_on_axis_mse(torch.tensor(B_coil), r=self.r, ntheta=self.ntheta)
        return loss

    def dfield_error(self):
        """
        Derivative of the field error w.r.t all coil coeffs,
            axis coefs, and etabar.

        return: SIMSOPT Derivative object containing the derivatives
            of the .J function with respect to the BiotSavart
            and Expansion DOFs.
        """
        self.qsc.calculate_or_cache()
        # Qsc field
        X_target, d_l_d_phi = self.qsc.downsample_axis(nphi=self.ntarget) # (3, ntarget), (ntarget)
        B_qsc = self.qsc.B_external_on_axis(r=self.r, ntheta=self.ntheta, X_target = X_target.T).T.detach().numpy() # (ntarget, 3)

        # coil field
        X_target_np = X_target.detach().numpy().T # (ntarget, 3)
        X_target_np = np.ascontiguousarray(X_target_np) # (ntarget, 3)
        self.bs.set_points(X_target_np)
        B_coil = self.bs.B() # (ntarget, 3)

        # compute dl
        dphi = np.diff(self.qsc.phi)[0]
        dl = (d_l_d_phi * dphi).detach().numpy().reshape((-1,1)) # (ntarget, 1)

        # derivative with respect to biot savart dofs
        dJ_by_dbs = self.bs.B_vjp((B_coil - B_qsc)*dl) # Derivative object

        """ Derivative w.r.t. axis coeffs """
        # this part of the derivative treats B_coil as a constant, independent of the axis
        loss = self.field_error()
        dloss_by_ddofs = self.qsc.total_derivative(loss) # list

        # derivative of B_coil(xyz(axis_coeffs)) term
        # self.qsc.zero_grad()
        dB_by_dX_bs = self.bs.dB_by_dX() # (ntarget, 3, 3)
        term21 = np.einsum("ji,jki->jk", ((B_coil - B_qsc) * dl), dB_by_dX_bs) # (ntarget, 3)

        dofs = self.qsc.get_dofs(as_tuple=True)
        term2 = torch.autograd.grad(X_target.T, dofs, grad_outputs=torch.tensor(term21), retain_graph=True, allow_unused=True) # tuple

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

    def J(self):
        """Compute the objective function, returning a float.

        Returns:
            float: objective function value.
        """
        return self.field_error().detach().numpy().item()
    
    @derivative_dec
    def dJ(self):
        """Compute the gradient of the objective function.

        Returns:
            array: gradient of the objective function as an np arrray.
        """
        return self.dfield_error()

class GradExternalFieldError(Optimizable):
    def __init__(self, bs, qsc, r, ntheta=128, ntarget=32):
        """        Integrated mean-squared error between the gradient of the external magnetic field on axis
        and the gradient of the Biot Savart field over the magnetic axis,
                Loss = (1/2) int |grad_B_external - grad_B_bs|**2 dl
        grad_B_external is computed by the virtual casing integral by integrating over a surface of
        radius r.

        Args:
            bs (BiotSavart object): Biot Savart object.
            qsc (Optimizable, Qsc): Optimizable Qsc object for computing external field.
            r (float): minor radius of the flux surface on which to take the virtual casing integral.
            ntheta (int, optional): number of theta quadrature points. Defaults to 128.
            ntarget (int, optional): number of target points on the magnetic axis at which to discretize
                the objective function integral. Defaults to 32.
        """
        self.bs = bs
        self.qsc = qsc
        self.r = r
        self.ntheta = ntheta
        self.ntarget = ntarget
        Optimizable.__init__(self, depends_on=[bs, qsc])

    def field_error(self):
        """Compute the objective function.

        Returns:
            tensor: float tensor with the objective value.
        """
        self.qsc.calculate_or_cache()
        # evaluate coil field
        X_target, _ = self.qsc.downsample_axis(nphi=self.ntarget) # (3, ntarget)
        X_target_np = X_target.detach().numpy().T # (ntarget, 3)
        X_target_np = np.ascontiguousarray(X_target_np) # (ntarget, 3)

        self.bs.set_points(X_target_np)
        grad_B_coil = self.bs.dB_by_dX().T # (3, 3, ntarget)

        # compute loss
        loss = self.qsc.grad_B_external_on_axis_mse(torch.tensor(grad_B_coil), r=self.r, ntheta=self.ntheta)
        return loss

    def dfield_error(self):
        """
        Derivative of the field error w.r.t all coil coeffs,
            axis coefs, and etabar.

        Returns:
            SIMSOPT Derivative object: containing the derivatives of the .field_error function 
            with respect to the BiotSavart and Qsc DOFs.
        """
        self.qsc.calculate_or_cache()
        # Qsc field
        X_target, d_l_d_phi = self.qsc.downsample_axis(nphi=self.ntarget) # (3, ntarget), (ntarget)
        grad_B_qsc = self.qsc.grad_B_external_on_axis(r=self.r, ntheta=self.ntheta, X_target = X_target.T)
        grad_B_qsc = grad_B_qsc.detach().numpy().T # (ntarget, 3, 3)

        # coil field
        X_target_np = X_target.detach().numpy().T # (ntarget, 3)
        X_target_np = np.ascontiguousarray(X_target_np) # (ntarget, 3)
        self.bs.set_points(X_target_np)
        grad_B_coil = self.bs.dB_by_dX() # (ntarget, 3, 3)

        # compute dl
        # dphi = np.diff(self.qsc.phi)[0]
        dphi = (2 * np.pi / self.qsc.nfp) / self.ntarget
        dl = d_l_d_phi.detach().numpy().reshape((-1,1,1)) * dphi

        # derivative with respect to biot savart dofs
        v = np.ones(3)
        vterm = (grad_B_coil - grad_B_qsc)*dl
        _, dJ_by_dbs = self.bs.B_and_dB_vjp(v, vterm) # Derivative object
        
        """ Derivative w.r.t. axis coeffs """
        # this part of the derivative treats B_coil as a constant, independent of the axis
        loss = self.field_error()
        dloss_by_ddofs = self.qsc.total_derivative(loss) # list

        # derivative of B_coil(xyz(axis_coeffs)) term
        d2B_by_dXdX_bs = self.bs.d2B_by_dXdX() # (ntarget, 3, 3, 3)
        term21 = np.einsum("ilj,ijkl->ik", (grad_B_coil - grad_B_qsc) * dl, d2B_by_dXdX_bs) # (ntarget, 3)
        dofs = self.qsc.get_dofs(as_tuple=True)
        term2 = torch.autograd.grad(X_target.T, dofs, grad_outputs=torch.tensor(term21), retain_graph=True, allow_unused=True) # tuple

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

    def J(self):
        """Compute the objective function, returning a float. This function can be used for
        optimization.

        Returns:
            float: objective function value.
        """
        return self.field_error().detach().numpy().item()
    
    @derivative_dec
    def dJ(self):
        """Compute the gradient of the objective function as an array (not tensor). This function
        can be used for optimization.

        Returns:
            array: gradient of the objective function as an np arrray.
        """
        return self.dfield_error()    

    
class IotaPenalty(Optimizable):
    def __init__(self, qsc, iota_target):
        """Penalty function
            1/2 * ((iota - iota_target) / iota_target)^2

        Args:
            qsc (Optimizable, Qsc):
            iota_target (float): target value of iota
        """
        self.qsc = qsc
        self.iota_target = iota_target
        Optimizable.__init__(self, depends_on=[qsc])

    def penalty(self):
        """Compute the objective function.

        Returns:
            tensor: float tensor with objective function value.
        """
        self.qsc.calculate_or_cache()
        loss = 0.5 * ((self.qsc.iota - self.iota_target) / self.iota_target)**2
        return loss
    
    def dpenalty(self):
        """Compute the gradient of the objective function.

        Returns:
            Derivative: Simsopt Derivative object.
        """
        self.qsc.calculate_or_cache()
        loss = self.penalty()
        dloss_by_ddofs = self.qsc.total_derivative(loss) # list

        # make a derivative object
        derivs_axis = np.zeros(0)
        for g in dloss_by_ddofs:
            derivs_axis = np.append(derivs_axis, g.detach().numpy())
        # arr = np.array([g.detach().numpy().flatten() for g in dloss_by_ddofs]) # array
        dJ_by_daxis = Derivative({self.qsc: derivs_axis})
        return dJ_by_daxis
    
    def J(self):
        """Compute the objective function, returning a float.

        Returns:
            float: objective function value.
        """
        return self.penalty().detach().numpy().item()
    
    @derivative_dec
    def dJ(self):
        """Compute the gradient of the objective function.

        Returns:
            array: gradient of the objective function as an np arrray.
        """
        return self.dpenalty()
    
class AxisLengthPenalty(Optimizable):
    def __init__(self, qsc, target_length):
        """Penalty function
            1/2 * ((length(axis) - target_length) / target_length)^2
        where the length is the total axis length (over all field periods).

        Args:
            qsc (Optimizable, Qsc):
            target_length (float): target value of length
        """
        self.qsc = qsc
        self.target_length = target_length
        Optimizable.__init__(self, depends_on=[qsc])
    
    def penalty(self):
        """Compute the objective function.

        Returns:
            tensor: float tensor with objective function value.
        """
        self.qsc.calculate_or_cache()
        loss = 0.5 * ((self.qsc.axis_length - self.target_length) / self.target_length)**2
        return loss
    
    def dpenalty(self):
        """Compute the gradient of the objective function.

        Returns:
            Derivative: Simsopt Derivative object.
        """
        self.qsc.calculate_or_cache()
        loss = self.penalty()
        dloss_by_ddofs = self.qsc.total_derivative(loss) # list

        # make a derivative object
        derivs_axis = np.zeros(0)
        for g in dloss_by_ddofs:
            derivs_axis = np.append(derivs_axis, g.detach().numpy())
        # arr = np.array([g.detach().numpy().flatten() for g in dloss_by_ddofs]) # array
        dJ_by_daxis = Derivative({self.qsc: derivs_axis})
        return dJ_by_daxis
    
    def J(self):
        """Compute the objective function, returning a float.

        Returns:
            float: objective function value.
        """
        return self.penalty().detach().numpy().item()
    
    @derivative_dec
    def dJ(self):
        """Compute the gradient of the objective function.

        Returns:
            array: gradient of the objective function as an np arrray.
        """
        return self.dpenalty()