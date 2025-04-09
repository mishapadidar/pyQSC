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
            of the .J function with respect to the BiotSavart
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
    
class GradFieldError(Optimizable):
    def __init__(self, bs, qsc):
        """Integrated mean-squared error between the gradient of the near axis field
        and the gradient of the Biot Savart magnetic field over the magnetic axis,
            Loss = (1/2) int |gradB - gradB_bs|**2 dl

        Args:
            bs (BiotSavart): Simsopt BiotSavart object.
            qsc (Qsc): Qsc object.
        """
        self.bs = bs
        self.qsc = qsc
        Optimizable.__init__(self, depends_on=[bs, qsc])

    def field_error(self):
        """
        Sum-of-squares error in the field:
            Loss = (1/2) int |gradB - gradB_bs|**2 dl/phi dphi
        where the integral is taken along the axis.
        """
        # evaluate coil field
        xyz = np.ascontiguousarray(self.qsc.XYZ0.detach().numpy().T) # (nphi, 3)
        self.bs.set_points(xyz)
        gradB_coil = self.bs.dB_by_dX().T # (3, 3, nphi)
        # compute loss
        loss = self.qsc.grad_B_tensor_cartesian_mse(torch.tensor(gradB_coil))
        return loss

    def dfield_error(self):
        """
        Derivative of the field error w.r.t all coil coeffs,
            axis coefs, and etabar.

        Returns:
            SIMSOPT Derivative object: containing the derivatives of the .field_error function 
            with respect to the BiotSavart and Qsc DOFs.
        """
        # Qsc field
        # X_target, d_l_d_phi = self.qsc.downsample_axis(nphi=self.ntarget) # (3, ntarget), (ntarget)
        # grad_B_qsc = self.qsc.grad_B_external_on_axis(r=self.r, ntheta=self.ntheta, nphi=self.nphi,
                                                    #   X_target = X_target.T)
        # grad_B_qsc = grad_B_qsc.detach().numpy().T # (ntarget, 3, 3)
        grad_B_qsc = self.qsc.grad_B_tensor_cartesian().detach().numpy().T # (ntarget, 3, 3)

        # coil field
        # X_target_np = X_target.detach().numpy().T # (ntarget, 3)
        # X_target_np = np.ascontiguousarray(X_target_np) # (ntarget, 3)
        xyz = np.ascontiguousarray(self.qsc.XYZ0.detach().numpy().T) # (nphi, 3)
        self.bs.set_points(xyz)
        # self.bs.set_points(X_target_np)
        grad_B_coil = self.bs.dB_by_dX() # (ntarget, 3, 3)

        # compute dl
        dphi = np.diff(self.qsc.phi)[0]
        d_l_d_phi = self.qsc.d_l_d_phi
        dl = (d_l_d_phi * dphi).detach().numpy().reshape((-1,1,1)) # (nphi, 1)

        # compute dl
        # dphi = np.diff(self.qsc.phi)[0]
        # dphi = (2 * np.pi / self.qsc.nfp) / self.ntarget
        # dl = d_l_d_phi.detach().numpy().reshape((-1,1,1)) * dphi

        # derivative with respect to biot savart dofs
        v = np.ones(3)
        vterm = (grad_B_coil - grad_B_qsc)*dl
        _, dJ_by_dbs = self.bs.B_and_dB_vjp(v, vterm) # Derivative object
        
        """ Derivative w.r.t. axis coeffs """
        # this part of the derivative treats B_coil as a constant, independent of the axis
        loss = self.field_error()
        dloss_by_ddofs = self.qsc.total_derivative(loss) # list

        # derivative of gradB_coil(xyz(axis_coeffs)) term
        d2B_by_dXdX_bs = self.bs.d2B_by_dXdX() # (ntarget, 3, 3, 3)
        term21 = np.einsum("ilj,ijkl->ik", (grad_B_coil - grad_B_qsc) * dl, d2B_by_dXdX_bs) # (ntarget, 3)
        dofs = self.qsc.get_dofs(as_tuple=True)
        X_target = self.qsc.XYZ0
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

class ExternalFieldError(Optimizable):
    def __init__(self, bs, qsc, r, ntheta=256, nphi=1024, ntarget=32):
        """Integrated mean-squared error between the Cartesian external magnetic field on axis
        and a target magnetic field over the magnetic axis,
                Loss = (1/2) int |B - B_target|**2 dl
        B is computed by the virtual casing integral by integrating over a surface of
        radius r. This class is a Simsopt Optimizable object.

        Args:
            bs (BiotSavart): Simsopt BiotSavart object.
            qsc (Qsc): Qsc object
            r (float): minor radius.
            ntheta (int, optional): Number of theta quadrature points for virtual casing integral.
                Defaults to 256.
            nphi (int, optional): Number of phi quadrature points for virtual casing integral.
                Defaults to 1024.
            ntarget (int, optional): Number of points on the magnetic axis at
                which to evaluate objective. Must be less than or equal to the number of points on axis.
                For spectral convergence choose ntarget to be a divisor of the number of points on the axis.
                Defaults to 32.
        """
        self.bs = bs
        self.qsc = qsc
        self.r = r
        self.ntheta = ntheta
        self.nphi = nphi
        self.ntarget = ntarget
        self.need_to_run_code = True
        Optimizable.__init__(self, depends_on=[bs, qsc])

    def recompute_bell(self, parent=None):
        """
        This function will get called any time any of the DOFs of the
        parent class change.
        need_to_run_code signifies the need to reevaluate the field_error method.
        """
        self.need_to_run_code = True
        return super().recompute_bell(parent)

    def field_error(self):
        """
        Sum-of-squares error in the virtual-casing field:
            (1/2) * int |B_axis - B_coil|^2 dl/dphi dphi
        where the integral is taken along the axis.

        Returns:
            tensor: float tensor with the objective value.
        """
        if not self.need_to_run_code:
            return self.loss
        
        # evaluate coil field
        X_target = self.qsc.subsample_axis_nodes(ntarget=self.ntarget)[0] # (3, ntarget)
        X_target_np = X_target.detach().numpy().T # (ntarget, 3)
        X_target_np = np.ascontiguousarray(X_target_np) # (ntarget, 3)

        self.bs.set_points(X_target_np)
        B_coil = self.bs.B().T # (3, ntarget)
        
        # compute loss
        loss = self.qsc.B_external_on_axis_mse(torch.tensor(B_coil), r=self.r, ntheta=self.ntheta, nphi = self.nphi)
        self.loss = loss
        self.need_to_run_code = False
        return loss

    def dfield_error(self):
        """
        Derivative of the field error w.r.t all coil coeffs,
            axis coefs, and etabar.

        return: SIMSOPT Derivative object containing the derivatives
            of the .J function with respect to the BiotSavart
            and Expansion DOFs.
        """
        # Qsc field
        X_target, d_l_d_phi, _ = self.qsc.subsample_axis_nodes(ntarget=self.ntarget) # (3, ntarget), (ntarget)
        B_qsc = self.qsc.B_external_on_axis_nodes(r=self.r, ntheta=self.ntheta, nphi=self.nphi, ntarget=self.ntarget).T.detach().numpy() # (ntarget, 3)

        # coil field
        X_target_np = X_target.detach().numpy().T # (ntarget, 3)
        X_target_np = np.ascontiguousarray(X_target_np) # (ntarget, 3)
        self.bs.set_points(X_target_np)
        B_coil = self.bs.B() # (ntarget, 3)

        # compute dl
        dphi = (2 * torch.pi / self.qsc.nfp) / self.ntarget
        dl = (d_l_d_phi * dphi).detach().numpy().reshape((-1,1)) # (ntarget, 1)

        # derivative with respect to biot savart dofs
        dJ_by_dbs = self.bs.B_vjp((B_coil - B_qsc)*dl) # Derivative object

        """ Derivative w.r.t. axis coeffs """
        # this part of the derivative treats B_coil as a constant, independent of the axis
        loss = torch.clone(self.field_error())
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
    def __init__(self, bs, qsc, r, ntheta=256, nphi=1024, ntarget=32):
        """        Integrated mean-squared error between the gradient of the external magnetic field on axis
        and the gradient of the Biot Savart field over the magnetic axis,
                Loss = (1/2) int |grad_B_external - grad_B_bs|**2 dl
        grad_B_external is computed by the virtual casing integral by integrating over a surface of
        radius r.

        Args:
            bs (BiotSavart): Simsopt BiotSavart object.
            qsc (Qsc): Qsc object
            r (float): minor radius.
            ntheta (int, optional): Number of theta quadrature points for virtual casing integral.
                Defaults to 256.
            nphi (int, optional): Number of phi quadrature points for virtual casing integral.
                Defaults to 1024.
            ntarget (int, optional): Number of points on the magnetic axis at
                which to evaluate objective. Must be less than or equal to the number of points on axis.
                For spectral convergence choose ntarget to be a divisor of the number of points on the axis.
                Defaults to 32.
        """
        self.bs = bs
        self.qsc = qsc
        self.r = r
        self.ntheta = ntheta
        self.nphi = nphi
        self.ntarget = ntarget
        self.need_to_run_code = True
        Optimizable.__init__(self, depends_on=[bs, qsc])

    def recompute_bell(self, parent=None):
        """
        This function will get called any time any of the DOFs of the
        parent class change.
        need_to_run_code signifies the need to reevaluate the field_error method.
        """
        self.need_to_run_code = True
        return super().recompute_bell(parent)

    def field_error(self):
        """Compute the objective function.

        Returns:
            tensor: float tensor with the objective value.
        """
        if not self.need_to_run_code:
            return self.loss
        
        # evaluate coil field
        X_target = self.qsc.subsample_axis_nodes(ntarget=self.ntarget)[0] # (3, ntarget)
        X_target_np = X_target.detach().numpy().T # (ntarget, 3)
        X_target_np = np.ascontiguousarray(X_target_np) # (ntarget, 3)

        self.bs.set_points(X_target_np)
        grad_B_coil = self.bs.dB_by_dX().T # (3, 3, ntarget)

        # compute loss
        loss = self.qsc.grad_B_external_on_axis_mse(torch.tensor(grad_B_coil), r=self.r,
                                                    ntheta=self.ntheta, nphi=self.nphi)
        self.loss = loss
        self.need_to_run_code = False
        return loss

    def dfield_error(self):
        """
        Derivative of the field error w.r.t all coil coeffs,
            axis coefs, and etabar.

        Returns:
            SIMSOPT Derivative object: containing the derivatives of the .field_error function 
            with respect to the BiotSavart and Qsc DOFs.
        """
        # Qsc field
        X_target, d_l_d_phi, _ = self.qsc.subsample_axis_nodes(ntarget=self.ntarget) # (3, ntarget), (ntarget)
        grad_B_qsc = self.qsc.grad_B_external_on_axis_nodes(r=self.r, ntheta=self.ntheta, nphi=self.nphi,
                                                      ntarget=self.ntarget)
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
        loss = torch.clone(self.field_error())
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
        loss = 0.5 * ((self.qsc.iota - self.iota_target) / self.iota_target)**2
        return loss
    
    def dpenalty(self):
        """Compute the gradient of the objective function.

        Returns:
            Derivative: Simsopt Derivative object.
        """
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
        loss = 0.5 * ((self.qsc.axis_length - self.target_length) / self.target_length)**2
        return loss
    
    def dpenalty(self):
        """Compute the gradient of the objective function.

        Returns:
            Derivative: Simsopt Derivative object.
        """
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
    
class LGradB(Optimizable):

    def __init__(self, qsc):
        """L-GradB objective from [1]
                J = I / (2 * L)
            where
                I = int |dB_by_dX|^2 dl
        and L is the axis length and the integral is taken over one field period.

        [1]: Mapping the space of quasisymmetric stellarators using optimized near-axis expansion,
            Landreman, (2022)
        Args:
            qsc (Optimizable, Qsc):
        """
        self.qsc = qsc
        # self.need_to_run_code = True
        Optimizable.__init__(self, depends_on=[qsc])

    # TODO: set up caching
    # def recompute_bell(self, parent=None):
    #     """
    #     This function will get called any time any of the DOFs of the
    #     parent class change.
    #     """
    #     self.need_to_run_code = True
    #     return super().recompute_bell(parent)

    def obj(self):
        """Compute the objective.

        Returns:
            tensor: float tensor of the objective value.
        """
        grad_B = self.qsc.grad_B_tensor_cartesian() # (3, 3, nphi)

        # compute dl
        dphi = np.diff(self.qsc.phi)[0]
        d_l_d_phi = self.qsc.d_l_d_phi # (nphi,)
        dl = d_l_d_phi * dphi

        axis_length = self.qsc.axis_length
        integral = torch.sum(torch.einsum('ijk->k', grad_B**2) * dl)
        J = integral / axis_length / 2
        return J

    def dobj(self):
        """Gradient of the obj function with respect to axis coefficients.

        Returns:
            Derivative: Simsopt Derivative object.
        """

        # compute derivative
        loss = self.obj()
        dloss_by_ddofs = self.qsc.total_derivative(loss) # list

        # make a derivative object
        derivs_axis = np.zeros(0)
        for g in dloss_by_ddofs:
            derivs_axis = np.append(derivs_axis, g.detach().numpy())

        dJ_by_daxis = Derivative({self.qsc: derivs_axis})
        return dJ_by_daxis
    
    def J(self):
        """Compute the objective function, returning a float.

        Returns:
            float: objective function value.
        """
        return self.obj().detach().numpy().item()
    
    @derivative_dec
    def dJ(self):
        """Compute the gradient of the objective function as a numpy array.

        Returns:
            array: gradient of the objective function as an np arrray.
        """
        return self.dobj()
    

class B20Penalty(Optimizable):

    def __init__(self, qsc):
        """f_B2 objective from [1]
                J = I / (2 * L)
            where
                I = int (B20 - mu)^2 dl
                mu = (1 / L) * int B20 dl.
            The objective measures the variance of B20. When B20 is constant, the stellarator is
            quasi-symmetric to second order. L is the axis length and the integral is taken over 
            one field period.

            Note that PyQSC must be run with order='r2' or order='r3' for this function to work.

        [1]: Mapping the space of quasisymmetric stellarators using optimized near-axis expansion,
            Landreman, (2022)
        Args:
            qsc (Optimizable, Qsc):
        """
        self.qsc = qsc
        # self.need_to_run_code = True
        Optimizable.__init__(self, depends_on=[qsc])

    # TODO: set up caching
    # def recompute_bell(self, parent=None):
    #     """
    #     This function will get called any time any of the DOFs of the
    #     parent class change.
    #     """
    #     self.need_to_run_code = True
    #     return super().recompute_bell(parent)

    def obj(self):
        """Compute the objective.

        Returns:
            tensor: float tensor of the objective value.
        """
        B20 = self.qsc.B20

        # compute dl
        dphi = np.diff(self.qsc.phi)[0]
        d_l_d_phi = self.qsc.d_l_d_phi # (nphi,)
        dl = d_l_d_phi * dphi
        axis_length = self.qsc.axis_length

        # compute the mean
        mu = torch.sum(B20 * dl) / axis_length

        # compute the variance
        integral = torch.sum((B20 - mu)**2 * dl)
        J = integral / axis_length / 2
        return J

    def dobj(self):
        """Gradient of the obj function with respect to axis coefficients.

        Returns:
            Derivative: Simsopt Derivative object.
        """

        # compute derivative
        loss = self.obj()
        dloss_by_ddofs = self.qsc.total_derivative(loss) # list

        # make a derivative object
        derivs_axis = np.zeros(0)
        for g in dloss_by_ddofs:
            derivs_axis = np.append(derivs_axis, g.detach().numpy())

        dJ_by_daxis = Derivative({self.qsc: derivs_axis})
        return dJ_by_daxis
    
    def J(self):
        """Compute the objective function, returning a float.

        Returns:
            float: objective function value.
        """
        return self.obj().detach().numpy().item()
    
    @derivative_dec
    def dJ(self):
        """Compute the gradient of the objective function as a numpy array.

        Returns:
            array: gradient of the objective function as an np arrray.
        """
        return self.dobj()

class MagneticWellPenalty(Optimizable):

    def __init__(self, qsc, well_target):
        """Magnetic Well Penalty objective from [1]
                J = max(0, d^2 V / dpsi^2 - W)^2 / W^2 
            where
                V is the volume,
                2 * pi * psi is the toroidal flux,
                W is the target value of the magnetic well.
            Typically, W is set to a negative value to provide a margin against instability.

            Note that PyQSC must be run with order='r2' or order='r3' for this function to work.

        [1]: Mapping the space of quasisymmetric stellarators using optimized near-axis expansion,
            Landreman, (2022)
        Args:
            qsc (Optimizable, Qsc)
            well_target (float): target value of the magnetic well.
        """
        self.qsc = qsc
        self.well_target = torch.tensor(well_target)
        # self.need_to_run_code = True
        Optimizable.__init__(self, depends_on=[qsc])

    # TODO: set up caching
    # def recompute_bell(self, parent=None):
    #     """
    #     This function will get called any time any of the DOFs of the
    #     parent class change.
    #     """
    #     self.need_to_run_code = True
    #     return super().recompute_bell(parent)

    def obj(self):
        """Compute the objective.

        Returns:
            tensor: float tensor of the objective value.
        """
        d2_volume_d_psi2 = self.qsc.d2_volume_d_psi2
        zero = torch.tensor(0)
        J = torch.max(zero, d2_volume_d_psi2 - self.well_target)**2 / (self.well_target**2)
        return J

    def dobj(self):
        """Gradient of the obj function with respect to axis coefficients.

        Returns:
            Derivative: Simsopt Derivative object.
        """

        # compute derivative
        loss = self.obj()
        dloss_by_ddofs = self.qsc.total_derivative(loss) # list

        # make a derivative object
        derivs_axis = np.zeros(0)
        for g in dloss_by_ddofs:
            derivs_axis = np.append(derivs_axis, g.detach().numpy())

        dJ_by_daxis = Derivative({self.qsc: derivs_axis})
        return dJ_by_daxis
    
    def J(self):
        """Compute the objective function, returning a float.

        Returns:
            float: objective function value.
        """
        return self.obj().detach().numpy().item()
    
    @derivative_dec
    def dJ(self):
        """Compute the gradient of the objective function as a numpy array.

        Returns:
            array: gradient of the objective function as an np arrray.
        """
        return self.dobj()

class AxisArcLengthVariation(Optimizable):

    def __init__(self, qsc):
        """Penalty on the variance of the arc length of the magnetic axis, 
            J = Var(d_l_d_phi)

        Args:
            qsc (Optimizable, Qsc)
        """
        self.qsc = qsc
        Optimizable.__init__(self, depends_on=[qsc])

    def obj(self):
        """Compute the objective.

        Returns:
            tensor: float tensor of the objective value.
        """
        d_l_d_phi = self.qsc.d_l_d_phi
        J = torch.var(d_l_d_phi)
        return J

    def dobj(self):
        """Gradient of the obj function with respect to axis coefficients.

        Returns:
            Derivative: Simsopt Derivative object.
        """

        # compute derivative
        loss = self.obj()
        dloss_by_ddofs = self.qsc.total_derivative(loss) # list

        # make a derivative object
        derivs_axis = np.zeros(0)
        for g in dloss_by_ddofs:
            derivs_axis = np.append(derivs_axis, g.detach().numpy())

        dJ_by_daxis = Derivative({self.qsc: derivs_axis})
        return dJ_by_daxis
    
    def J(self):
        """Compute the objective function, returning a float.

        Returns:
            float: objective function value.
        """
        return self.obj().detach().numpy().item()
    
    @derivative_dec
    def dJ(self):
        """Compute the gradient of the objective function as a numpy array.

        Returns:
            array: gradient of the objective function as an np arrray.
        """
        return self.dobj()