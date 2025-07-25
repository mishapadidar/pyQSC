"""
This module contains the top-level routines for the quasisymmetric
stellarator construction.
"""

import logging
import numpy as np
from scipy.io import netcdf
#from numba import jit
import torch
import numpy as np

# TODO: only modify the classes datatype.
# set 64-bit default
torch.set_default_dtype(torch.float64)

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Qsc(torch.nn.Module):
    """
    This is the main class for representing the quasisymmetric
    stellarator construction.
    """
    
    # Import methods that are defined in separate files:
    from .init_axis import init_axis, convert_to_spline
    from .calculate_r1 import _residual, _jacobian, solve_sigma_equation, \
        _determine_helicity, r1_diagnostics, dresidual_by_ddof_vjp, dresidual_vac_by_ddof_vjp
    from .grad_B_tensor import (calculate_grad_B_tensor, calculate_grad_grad_B_tensor,
        Bfield_cylindrical, Bfield_cartesian, grad_B_tensor_cartesian,
        grad_grad_B_tensor_cylindrical, grad_grad_B_tensor_cartesian,
        calculate_grad_B_tensor_vac, calculate_grad_grad_B_tensor_vac)
    from .calculate_r2 import calculate_r2, calculate_r2_vac
    from .calculate_r3 import calculate_r3, calculate_shear
    from .geo import (surface, dsurface_by_dvarphi, dsurface_by_dtheta,
                      dsurface_by_dr, surface_normal, jacobian, d2surface_by_dthetatheta,
                      surface_theta_curvature, surface_area, surface_area_element)
    from .mercier import mercier
    from .r_singularity import calculate_r_singularity
    from .plot import plot, plot_boundary, get_boundary, B_fieldline, B_contour, plot_axis, flux_tube
    from .Frenet_to_cylindrical import Frenet_to_cylindrical, to_RZ
    from .to_vmec import to_vmec
    from .util import B_mag
    from .virtual_casing import (B_external_on_axis, B_taylor, B_external_on_axis_taylor,
                                 grad_B_external_on_axis, build_virtual_casing_interpolants,
                                 B_external_on_axis_nodes, grad_B_external_on_axis_nodes,
                                 B_external_on_axis_split, build_virtual_casing_interpolants_split,
                                 grad_B_external_on_axis_split, curl_taylor, divergence_taylor)
    from .configurations import from_paper, configurations
    from .objectives import (Bfield_axis_mse, grad_B_tensor_cartesian_mse, total_derivative,
                             B_external_on_axis_mse, downsample_axis, subsample_axis_nodes, grad_B_external_on_axis_mse,
                             surface_integral)
    
    def __init__(self, rc, zs, rs=[], zc=[], nfp=1, etabar=1., sigma0=0., B0=1.,
                 I2=0., sG=1, spsi=1, nphi=61, B2s=0., B2c=0., p2=0., order="r1"):
        """Initialize Qsc

        Args:
            rc (array-like): list of all rc parameters of the axis.
            zs (array-like): list of all zs parameters of the axis.
            rs (array-like, optional): list of all rs parameters of the axis. Defaults to [].
            zc (array-like, optional): list of all zc parameters of the axis.. Defaults to [].
            nfp (int, optional): number of field periods. Defaults to 1.
            etabar (float, optional): etabar parameter. Defaults to 1..
            sigma0 (float, optional): sigma0 parameter. Defaults to 0..
            B0 (float, optional): axis field strength. Defaults to 1..
            I2 (float, optional): I2 current parameter. Defaults to 0..
            sG (int, optional): sign of G. Defaults to 1.
            spsi (int, optional): sign of psi. Defaults to 1.
            nphi (int, optional): number of quadrature points on the axis. Defaults to 61.
            B2s (float, optional): Defaults to 0..
            B2c (float, optional): Defaults to 0..
            p2 (float, optional): p2 pressure parameter. Defaults to 0..
            order (str, optional): expansion order. Defaults to "r1".
        """
        super().__init__()

        # First, force {rc, zs, rs, zc} to have the same length, for
        # simplicity.
        nfourier = torch.max(torch.tensor([len(rc), len(zs), len(rs), len(zc)]))
        self.nfourier = nfourier
        self.rc = torch.nn.Parameter(torch.zeros(nfourier), requires_grad=True)
        self.zs = torch.nn.Parameter(torch.zeros(nfourier), requires_grad=True)
        self.rs = torch.nn.Parameter(torch.zeros(nfourier), requires_grad=True)
        self.zc = torch.nn.Parameter(torch.zeros(nfourier), requires_grad=True)
        # self.rc = torch.zeros(nfourier, requires_grad=True)
        # self.zs = torch.zeros(nfourier, requires_grad=True)
        # self.rs = torch.zeros(nfourier, requires_grad=True)
        # self.zc = torch.zeros(nfourier, requires_grad=True)
        self.rc[:len(rc)].data += torch.clone(torch.tensor(rc)).detach()
        self.zs[:len(zs)].data += torch.clone(torch.tensor(zs)).detach()
        self.rs[:len(rs)].data += torch.clone(torch.tensor(rs)).detach()
        self.zc[:len(zc)].data += torch.clone(torch.tensor(zc)).detach()

        # Force nphi to be odd:
        if np.mod(nphi, 2) == 0:
            nphi += 1

        if sG != 1 and sG != -1:
            raise ValueError('sG must be +1 or -1')
        
        if spsi != 1 and spsi != -1:
            raise ValueError('spsi must be +1 or -1')

        self.nfp = nfp
        self.etabar = torch.nn.Parameter(torch.tensor(etabar), requires_grad=True)
        self.B2c = torch.nn.Parameter(torch.tensor(B2c), requires_grad=True)
        self.B2s = torch.nn.Parameter(torch.tensor(B2s), requires_grad=True)
        self.sigma0 =  sigma0
        self.B0 =  B0
        self.I2 =  torch.nn.Parameter(torch.tensor(I2), requires_grad=True)
        self.p2 =  torch.nn.Parameter(torch.tensor(p2), requires_grad=True)
        self.sG = sG
        self.spsi = spsi
        self.nphi = nphi
        self.order = order
        self.min_R0_threshold = 0.3
        self._set_names()

        self.calculate()

    # def change_nfourier(self, nfourier_new):
    #     """
    #     Resize the arrays of Fourier amplitudes. You can either increase
    #     or decrease nfourier.
    #     """
    #     rc_old = self.rc
    #     rs_old = self.rs
    #     zc_old = self.zc
    #     zs_old = self.zs
    #     index = torch.min(torch.tensor([self.nfourier, nfourier_new]))
    #     self.rc = torch.zeros(nfourier_new, requires_grad=True)
    #     self.rs = torch.zeros(nfourier_new, requires_grad=True)
    #     self.zc = torch.zeros(nfourier_new, requires_grad=True)
    #     self.zs = torch.zeros(nfourier_new, requires_grad=True)
    #     self.rc[:index] = rc_old[:index]
    #     self.rs[:index] = rs_old[:index]
    #     self.zc[:index] = zc_old[:index]
    #     self.zs[:index] = zs_old[:index]
    #     nfourier_old = self.nfourier
    #     self.nfourier = nfourier_new
    #     self._set_names()
    #     # No need to recalculate if we increased the Fourier
    #     # resolution, only if we decreased it.
    #     if nfourier_new < nfourier_old:
    #         self.calculate()

    def calculate(self):
        """
        Driver for the main calculations.
        """
        self.init_axis()
        self.solve_sigma_equation()
        self.r1_diagnostics()
        if self.order != 'r1':
            self.calculate_r2()
            self.calculate_r2_vac()
            if self.order == 'r3':
                self.calculate_r3()
        self.clear_cache()

    def clear_cache(self):
        """
        Clear the cached values for the virtual casing and other.
        """
        # clear the cached values
        self.build_virtual_casing_interpolants.cache_clear()
        self.build_virtual_casing_interpolants_split.cache_clear()
        self.B_external_on_axis_nodes.cache_clear()
        self.grad_B_external_on_axis_nodes.cache_clear()
    
    def get_dofs(self, as_tuple=False):
        """
        Return a 1D numpy vector of all possible optimizable
        degrees-of-freedom, for simsopt.
        """
        if as_tuple:
            dofs = (self.rc, self.zs, self.rs, self.zc, self.etabar, self.B2s, self.B2c, self.p2, self.I2)
            return dofs
        else:
            dofs = torch.concatenate((self.rc, self.zs, self.rs, self.zc,
                                      torch.tensor([self.etabar, self.B2s, self.B2c, self.p2, self.I2])
                                      ))
            return dofs.detach().numpy()

    def set_dofs(self, x):
        """ For interaction with simsopt, set the optimizable degrees of
        freedom from a 1D numpy vector.

        Args:
            x (array): numpy array of dofs.
        """
        self.ndofs = self.nfourier * 4 + 5
        assert len(x) == self.ndofs
        self.rc.data = torch.clone(torch.tensor(x[self.nfourier * 0 : self.nfourier * 1])).detach()
        self.zs.data = torch.clone(torch.tensor(x[self.nfourier * 1 : self.nfourier * 2])).detach()
        self.rs.data = torch.clone(torch.tensor(x[self.nfourier * 2 : self.nfourier * 3])).detach()
        self.zc.data = torch.clone(torch.tensor(x[self.nfourier * 3 : self.nfourier * 4])).detach()
        self.etabar.data = torch.clone(torch.tensor(x[self.nfourier * 4 + 0])).detach()
        self.B2s.data = torch.clone(torch.tensor(x[self.nfourier * 4 + 1])).detach()
        self.B2c.data = torch.clone(torch.tensor(x[self.nfourier * 4 + 2])).detach()
        self.p2.data = torch.clone(torch.tensor(x[self.nfourier * 4 + 3])).detach()
        self.I2.data = torch.clone(torch.tensor(x[self.nfourier * 4 + 4])).detach()

        self.calculate()
        logger.info('set_dofs called with x={}. Now iota={}, elongation={}'.format(x, self.iota, self.max_elongation))
        
    def _set_names(self):
        """
        For simsopt, sets the list of names for each degree of freedom.
        """
        names = []
        names += ['rc({})'.format(j) for j in range(self.nfourier)]
        names += ['zs({})'.format(j) for j in range(self.nfourier)]
        names += ['rs({})'.format(j) for j in range(self.nfourier)]
        names += ['zc({})'.format(j) for j in range(self.nfourier)]
        names += ['etabar','B2s', 'B2c', 'p2', 'I2']
        self.names = names

    # @classmethod
    # def from_cxx(cls, filename):
    #     """
    #     Load a configuration from a ``qsc_out.<extension>.nc`` output file
    #     that was generated by the C++ version of QSC. Almost all the
    #     data will be taken from the output file, over-writing any
    #     calculations done in python when the new Qsc object is
    #     created.
    #     """
    #     def to_string(nc_str):
    #         """ Convert a string from the netcdf binary format to a python string. """
    #         temp = [c.decode('UTF-8') for c in nc_str]
    #         return (''.join(temp)).strip()
        
    #     f = netcdf.netcdf_file(filename, mmap=False)
    #     nfp = f.variables['nfp'][()]
    #     nphi = f.variables['nphi'][()]
    #     rc = f.variables['R0c'][()]
    #     rs = f.variables['R0s'][()]
    #     zc = f.variables['Z0c'][()]
    #     zs = f.variables['Z0s'][()]
    #     I2 = f.variables['I2'][()]
    #     B0 = f.variables['B0'][()]
    #     spsi = f.variables['spsi'][()]
    #     sG = f.variables['sG'][()]
    #     etabar = f.variables['eta_bar'][()]
    #     sigma0 = f.variables['sigma0'][()]
    #     order_r_option = to_string(f.variables['order_r_option'][()])
    #     if order_r_option == 'r2.1':
    #         order_r_option = 'r3'
    #     if order_r_option == 'r1':
    #         p2 = 0.0
    #         B2c = 0.0
    #         B2s = 0.0
    #     else:
    #         p2 = f.variables['p2'][()]
    #         B2c = f.variables['B2c'][()]
    #         B2s = f.variables['B2s'][()]

    #     q = cls(nfp=nfp, nphi=nphi, rc=rc, rs=rs, zc=zc, zs=zs,
    #             B0=B0, sG=sG, spsi=spsi,
    #             etabar=etabar, sigma0=sigma0, I2=I2, p2=p2, B2c=B2c, B2s=B2s, order=order_r_option)
        
    #     def read(name, cxx_name=None):
    #         if cxx_name is None: cxx_name = name
    #         setattr(q, name, f.variables[cxx_name][()])

    #     [read(v) for v in ['R0', 'Z0', 'R0p', 'Z0p', 'R0pp', 'Z0pp', 'R0ppp', 'Z0ppp',
    #                        'sigma', 'curvature', 'torsion', 'X1c', 'Y1c', 'Y1s', 'elongation']]
    #     if order_r_option != 'r1':
    #         [read(v) for v in ['X20', 'X2c', 'X2s', 'Y20', 'Y2c', 'Y2s', 'Z20', 'Z2c', 'Z2s', 'B20']]
    #         if order_r_option != 'r2':
    #             [read(v) for v in ['X3c1', 'Y3c1', 'Y3s1']]
                    
    #     f.close()
    #     return q
        
    # def min_R0_penalty(self):
    #     """
    #     This function can be used in optimization to penalize situations
    #     in which min(R0) < min_R0_constraint.
    #     """
    #     return torch.max((0, self.min_R0_threshold - self.min_R0)) ** 2
        
