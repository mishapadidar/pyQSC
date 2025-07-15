#!/usr/bin/env python3

"""
Metrics for optimization.
"""

import logging
import numpy as np
import torch
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt._core.json import GSONDecoder
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
    
    def as_dict(self, serial_objs_dict=None) -> dict:
        """Used by the simsopt save to save a QscOptimizable object.
        Args:
            serial_objs_dict: dictionary, used by the :obj:`GSONDecoder` recursive routine to save simsopt objects.

        Returns:
            dict: containing the information needed to serialize the object.
        """
        d = super().as_dict(serial_objs_dict=serial_objs_dict)

        # add the non-DOF attributes needed to initialize a QSC object.
        d["nfp"] = self.nfp
        d["sigma0"] = self.sigma0
        d["B0"] = self.B0
        d["sG"] = self.sG
        d["spsi"] = self.spsi
        d["nphi"] = self.nphi
        d["order"] = self.order
        return d
    
    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        """Used by the json decoder to load a QscOptimizable object
        
        Args:
            d (dict): Contains the dofs and other class attributes needed for initialization.
            serial_objs_dict: dictionary, used by the :obj:`GSONDecoder` recursive routine to load simsopt objects.
            recon_objs: dictionary, used by the :obj:`GSONDecoder` recursive routine to load simsopt objects.
        
        Returns:
            An instance of the QscOptimizable class, as described by the dictionary d
        """
        decoder = GSONDecoder()

        # get the dofs
        dofs = d.pop('dofs')
        dofs =  decoder.process_decoded(dofs, serial_objs_dict, recon_objs)
        x = dofs._x
        names = dofs._names
        rc = [x[ii] for ii, nn in enumerate(names) if 'rc' in nn]
        rs = [x[ii] for ii, nn in enumerate(names) if 'rs' in nn]
        zc = [x[ii] for ii, nn in enumerate(names) if 'zc' in nn]
        zs = [x[ii] for ii, nn in enumerate(names) if 'zs' in nn]
        etabar = x[ names.index('etabar')]
        I2 =  x[ names.index('I2')]
        p2 =  x[ names.index('p2')]
        B2s =  x[ names.index('B2s')]
        B2c =  x[ names.index('B2c')]

        # initialize the class
        out = cls(rc = rc, rs=rs, zc = zc, zs=zs, etabar = etabar, 
                  I2 = I2, p2 = p2, B2s = B2s, B2c = B2c, 
                  **d)

        return out
    
    def get_scale(self, **kwargs):
        """
        Construct an array (not tensor) of scales for each unfixed degree of freedom,
        ordered by the degrees of freedom. The scale can be used to normalize
        the degrees of freedom for optimization to approximately unit scale,

        Example:
            stel = Qsc.from_paper('preceise QA')
            scale = stel.get_scale(**{'p2':5.0, 'zs(1)':7.0})
            x_scaled = stel.x / scale # scaled DOFs

        Args:
            **kwargs: the scale for each degree of freedom can be passed as a keyword argument,
                e.g. rc(0)=0.1, zs(2)=0.01, etabar=0.5, etc. If a degree of freedom is not passed,
                a default scale is used: for Fourier modes with mode number m, the default scale is exp(-m),
                and for the non-Fourier parameters (etabar, B2s, B2c, p2, I2) the default scale is 1.0.

        Returns:
            array: array of scales for each degree of freedom, in the same order as the (free) DOFs.
        """
        scale = []
        for nn in self.names:
            if self.is_free(nn):
                if ('rc' in nn) or ('zs' in nn) or ('rs' in nn) or ('zc' in nn):
                    # get the mode number
                    mode_number = int(nn.split('(')[1].split(')')[0])
                    scale.append(kwargs.pop(nn, np.exp(- mode_number)))
                else:
                    scale.append(kwargs.pop(nn, 1.0))
        return np.array(scale)


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
        
        # dont compute anything if axis dofs are all fixed
        if np.any(self.qsc.dofs_free_status):

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
        else:
            dJ = dJ_by_dbs

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
        # dont compute anything if axis dofs are all fixed
        if np.any(self.qsc.dofs_free_status):
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
        else:
            dJ = dJ_by_dbs

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
    
class GradBPenalty(Optimizable):

    def __init__(self, qsc):
        """GradB penalty from [1]
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
    
class GradGradBPenalty(Optimizable):

    def __init__(self, qsc):
        """GradGradB penalty from [1]
                J = I / (2 * L)
            where
                I = int |d2B_by_dX^2|^2 dl
        and L is the axis length and the integral is taken over one field period.

        [1]: Mapping the space of quasisymmetric stellarators using optimized near-axis expansion,
            Landreman, (2022)
        Args:
            qsc (Optimizable, Qsc):
        """
        self.qsc = qsc
        Optimizable.__init__(self, depends_on=[qsc])

    def obj(self):
        """Compute the objective.

        Returns:
            tensor: float tensor of the objective value.
        """
        grad_grad_B = self.qsc.grad_grad_B_tensor_cartesian() # (3, 3, 3, nphi)

        # compute dl
        dphi = np.diff(self.qsc.phi)[0]
        d_l_d_phi = self.qsc.d_l_d_phi # (nphi,)
        dl = d_l_d_phi * dphi

        axis_length = self.qsc.axis_length
        norm_squared = torch.sum(grad_grad_B**2, dim=[0,1,2]) # (nphi, )
        integral = torch.sum(norm_squared * dl)
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

class SurfaceSelfIntersectionPenalty(Optimizable):

    def __init__(self, qsc, r, ntheta=64, tol=5e-2):
        """Penalty on the self-intersection of flux surfaces.
        Flux surfaces self-intersect when the jacobian of the coordinate
        transformation, sqrtg, becomes singular. This penalty attempts to enforce
        the constraint,
            sqrtg(theta_ij, phi_ij) >= tol,
        at each quadrature point (theta_ij, phi_ij) on a flux surface. The penalty function is,
            J = mean_ij (max(0, tol - sqrtg_ij))**2 / tol.
        We dont divide by tol^2 since tol is a small number.

        This class is a Simsopt Optimizable object.

        Args:
            qsc (Optimizable, Qsc): Optimizable Qsc object.
            r (float): minor radius of the flux surface to evaluate.
            ntheta (int, optional): Number of theta quadrature points at which to evaluate objective
                on the flux surface. Defaults to 64.
            tol (float, optional): tolerance for the jacobian. Defaults to 5e-2.
        """
        self.qsc = qsc
        self.r = r
        self.ntheta = ntheta
        self.tol = torch.tensor(tol)
        Optimizable.__init__(self, depends_on=[qsc])

    def obj(self):
        """Compute the objective.

        Returns:
            tensor: float tensor of the objective value.
        """
        sqrtg = self.qsc.jacobian(r = self.r, ntheta=self.ntheta)
        violation = torch.max(torch.tensor(0.0), self.tol - sqrtg)
        J = torch.mean(violation**2) / self.tol
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
    
class PressurePenalty(Optimizable):
    def __init__(self, qsc, p2_target=-1e6):
        """Penalize deviation from a desired pressure profile
            1/2 * ((p2 - p2_target) / p2_target)^2
        When using this objective, p2 should be an unfixed degree of freedom.

        Args:
            qsc (Optimizable, Qsc):
            p2_target (float): target value of p2. Defaults to -1e6.
        """
        self.qsc = qsc
        self.p2_target = p2_target
        Optimizable.__init__(self, depends_on=[qsc])

    def obj(self):
        """Compute the objective function.

        Returns:
            tensor: float tensor with objective function value.
        """
        loss = 0.5 * ((self.qsc.p2 - self.p2_target) / self.p2_target)**2
        return loss
    
    def dobj(self):
        """Compute the gradient of the objective function.

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
        """Compute the gradient of the objective function.

        Returns:
            array: gradient of the objective function as an np arrray.
        """
        return self.dobj()


class CurveAxisDistancePenalty(Optimizable):
    def __init__(self, curve, qsc, minimum_distance):
        """
        A penalty function that penalizes the distance between a Simsopt Curve and the axis.
        This is useful for keeping coils far away from the axis. This can be used in place
        of a coil plasma distance penalty.

        curve (Curve): A Simsopt Curve object
        qsc (QscOptimizable): a QscOptimizable object
        """
        self.curve = curve
        self.qsc = qsc
        self.minimum_distance = minimum_distance
        Optimizable.__init__(self, depends_on=[curve, qsc])

    def shortest_distance(self):
        """Compute the shortest distance between the curve and the axis.

        Returns:
            tensor: float tensor with the shortest distance.
        """
        # get axis and curve positions
        xyz_axis = self.qsc.XYZ0.T # (nphi, 3)
        xyz_curve = torch.tensor(self.curve.gamma()) # (ncurve, 3)

        # compute pairwise distances
        dist = torch.linalg.norm(xyz_axis[:, None, :] - xyz_curve[None, :, :], dim=-1) # (nphi, ncurve)
        return torch.min(dist)
    
    def obj(self):
        """
        Compute the penalty function.
            J = int_{curve} int_{axis} 0.5 * max(0, d_min - ||r(c) - s(l)||)^2 dl dc
        where
            r(c) is the position vector of the curve,
            s(l) is the position vector of the axis,
            d_min is the minimum distance tolerance between the curve and the axis.
        """
        # get axis and curve positions
        xyz_axis = self.qsc.XYZ0.T # (nphi, 3)
        xyz_curve = torch.tensor(self.curve.gamma()) # (ncurve, 3)

        # compute pairwise distances
        dist = torch.linalg.norm(xyz_curve[None, :, :] - xyz_axis[:, None, :], dim=-1) # (nphi, ncurve)
        integrand = 0.5 * torch.maximum(self.minimum_distance - dist, torch.tensor(0))**2

        # line elements
        dphi = torch.diff(self.qsc.phi)[0]
        d_l_d_phi = self.qsc.d_l_d_phi
        dl_axis = d_l_d_phi * dphi # (nphi,)
        dphi_curve = np.diff(self.curve.quadpoints)[0]
        d_l_d_phi_curve = np.linalg.norm(self.curve.gammadash(), axis=-1) # (ncurve,)
        dl_curve = torch.tensor(d_l_d_phi_curve * dphi_curve) # (ncurve,)

        # integrate
        J = torch.sum(integrand * dl_axis[:, None] * dl_curve[None, :])
        return J

    def dobj(self):
        """
        Derivative of obj w.r.t all coil dofs and axis dofs.

        return: SIMSOPT Derivative object containing the derivatives
            of the .obj function with respect to the curve
            and qsc DOFs.
        """
        # get axis and curve positions
        xyz_axis = self.qsc.XYZ0.T.detach().numpy() # (nphi, 3)
        xyz_curve = self.curve.gamma() # (ncurve, 3)

        # compute pairwise distances
        diff = xyz_curve[None, :, :] - xyz_axis[:, None, :] # (nphi, ncurve, 3)
        dist = np.linalg.norm(diff, axis=-1) # (nphi, ncurve)
        integrand = 0.5 * np.maximum(self.minimum_distance - dist, 0.0)**2

        # line elements
        dphi = np.diff(self.qsc.phi.detach().numpy())[0]
        d_l_d_phi = self.qsc.d_l_d_phi.detach().numpy() # (nphi,)
        dl_axis = (d_l_d_phi * dphi)[:, None] # (nphi,1)
        dphi_curve = np.diff(self.curve.quadpoints)[0]
        d_l_d_phi_curve = np.linalg.norm(self.curve.gammadash(), axis=-1) # (ncurve,)
        dl_curve = (d_l_d_phi_curve * dphi_curve)[:, None] # (ncurve,1)

        # derivative of the integrand wrt curve dofs
        inner = - (np.maximum(self.minimum_distance - dist, 0.0) * dl_axis / dist)[:,:,None] * diff
        dintegrand_by_dcurve = np.sum(inner, axis=0) # (ncurve, 3)
        dintegrand_by_dcurve = self.curve.dgamma_by_dcoeff_vjp(dintegrand_by_dcurve * dl_curve)

        # derivative of the arc length wrt curve dofs
        inner = np.sum(integrand * dl_axis * dphi_curve, axis=0) # (ncurve,)
        dl_by_dcurve_term = self.curve.dincremental_arclength_by_dcoeff_vjp(inner)

        # Derivative object
        dJ_by_dcurve = dintegrand_by_dcurve + dl_by_dcurve_term

        """ Derivative w.r.t. axis dofs """
        loss = self.obj()
        dloss_by_ddofs = self.qsc.total_derivative(loss) # list

        # make a derivative object
        derivs_axis = np.zeros(0)
        for g in dloss_by_ddofs:
            derivs_axis = np.append(derivs_axis, g.detach().numpy())

        # make a derivative object
        dJ_by_daxis = Derivative({self.qsc: derivs_axis})

        dJ = dJ_by_daxis + dJ_by_dcurve

        return dJ
    
    def J(self):
        """Compute the objective function, returning a float.

        Returns:
            float: objective function value.
        """
        return self.obj().detach().numpy().item()
    
    @derivative_dec
    def dJ(self):
        """Compute the gradient of the objective function.

        Returns:
            array: gradient of the objective function as an np arrray.
        """
        return self.dobj()
    
class ThetaCurvaturePenalty(Optimizable):
    def __init__(self, qsc, r, ntheta=32, kappa_target=0.0):
        """Penalize the curvature of a flux surface in the theta direction.
        This penalty reflects the inequality constraint,
            kappa_theta <= kappa_target,
        pointwise on a flux surface, where kappa_theta is the curvature 
        of the flux surface in the theta direction. The penalty is,
            J = (1/A) * int max(0, kappa_theta - kappa_target)^2 dA
        where A is the area of the flux surface.

        Args:
            qsc (QscOptimizable): a QscOptimizable object
            kappa_target (float): target value of kappa. Defaults to 0.
        """
        self.qsc = qsc
        self.r = r
        self.ntheta = ntheta
        self.kappa_target = kappa_target
        Optimizable.__init__(self, depends_on=[qsc])

    def obj(self):
        """Compute the objective function.

        Returns:
            tensor: float tensor with objective function value.
        """
        area = self.qsc.surface_area(self.r, ntheta=self.ntheta)
        kappa = self.qsc.surface_theta_curvature(r=self.r, ntheta=self.ntheta) # (nphi, ntheta)
        integrand = torch.maximum(torch.tensor(0.0), kappa - self.kappa_target)**2
        loss = self.qsc.surface_integral(integrand, self.r) / area
        return loss
    
    def dobj(self):
        """Compute the gradient of the objective function.

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
        """Compute the gradient of the objective function.

        Returns:
            array: gradient of the objective function as an np arrray.
        """
        return self.dobj()