"""
This module contains the functions for solving the sigma equation
and computing diagnostics of the O(r^1) solution.
"""

import logging
import numpy as np
import torch
from .util import fourier_minimum
from .newton import newton
from torch.autograd.functional import jacobian
from functools import partial

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _residual(self, x):
    """
    Residual in the sigma equation, used for Newton's method.  x is
    the state vector, corresponding to sigma on the phi grid,
    except that the first element of x is actually iota.
    """
    sigma = torch.clone(x)
    sigma[0] = self.sigma0
    iota = x[0]
    r = torch.matmul(self.d_d_varphi, sigma) \
        + (iota + self.helicity * self.nfp) * \
        (self.etabar_squared_over_curvature_squared * self.etabar_squared_over_curvature_squared + 1 + sigma * sigma) \
        - 2 * self.etabar_squared_over_curvature_squared * (-self.spsi * self.torsion + self.I2 / self.B0) * self.G0 / self.B0
    
    #logger.debug("_residual called with x={}, r={}".format(x, r))
    return r

def _jacobian(self, x):
    """
    Compute the Jacobian matrix for solving the sigma equation. x is
    the state vector, corresponding to sigma on the phi grid,
    except that the first element of x is actually iota.
    """
    sigma = torch.clone(x)
    sigma[0] = self.sigma0
    iota = x[0]

    # d (Riccati equation) / d sigma:
    # For convenience we will fill all the columns now, and re-write the first column in a moment.
    jac = torch.clone(self.d_d_varphi)
    for j in range(self.nphi):
        jac[j, j] += (iota + self.helicity * self.nfp) * 2 * sigma[j]

    # d (Riccati equation) / d iota:
    jac[:, 0] = self.etabar_squared_over_curvature_squared * self.etabar_squared_over_curvature_squared + 1 + sigma * sigma

    #logger.debug("_jacobian called with x={}, jac={}".format(x, jac))
    return jac

def solve_sigma_equation(self):
    """
    Solve the sigma equation.
    """
    x0 = self.sigma0 * torch.ones(self.nphi)
    x0[0] = 0 # Initial guess for iota
    """
    soln = scipy.optimize.root(self._residual, x0, jac=self._jacobian, method='lm')
    self.iota = soln.x[0]
    self.sigma = torch.clone(soln.x)
    self.sigma[0] = self.sigma0
    """
    sol = newton(self._residual, x0, jac=self._jacobian) 

    # detach so we DOFS dont differentiate through sigma/iota.
    sigma = torch.clone(sol).detach()
    iota = torch.clone(sol[0]).detach()
    sigma[0] = self.sigma0

    # set sigma/iota to be variables we can differentiate through later
    self.sigma = torch.nn.Parameter(sigma, requires_grad=True)
    self.iota =  torch.nn.Parameter(iota, requires_grad=True)
    self.iotaN = self.iota + self.helicity * self.nfp

def dresidual_by_ddof_vjp(self, v):
    """
    Differentiate the solution of the sigma equation with respect to the degrees of
    freedom.

    The residual satisfies
        r(z(x), x) = 0
    where x are the dofs and z(x) = [iota(x), sigma(x)]. The derivative system at a solution is,
        dr/dz * dz/dx = - dr/dx,
    where we aim to solve for the jacobian, dz/dx. 
    
    For optimization we only need know,
        transpose(dz/dx) * v
    for a probe vector v, so we solve for this instead. This has the added benefit of 
    playing well with torch vector jacobian products.
    
    The solution to the derivative system is,
            dz/dx = - inv(dr/dz) * dr/dx.
    Transposing the system and right-multiplying by a vector v of length nphi,
        transpose(dz/dx) * v = - transpose(dr/dx) * inv(tranpose(dr/dz)) * v.
    If we define lambda via,
        tranpose(dr/dz) * lambda = - v,                           (1)
    then we can write our solution as, 
        transpose(dz/dx) * v = transpose(dr/dx) * lambda.         (2)

    Hence computing the derivative has two steps: solve (1) for lambda, use autodiff to
    compute transpose(dr/dx) * lambda with (2).

    For our problem, z(x) is the function
        z(x) = [iota(x), sigma_1(x), ..., sigma_nphi(x)],
    since sigma_0 is a fixed value. This means that dz/dx has the structure, 
        dz/dx = [[diota/dx], [dsigma_1/dx], ..., [dsigma_nphi/dx]]
    where rows are gradients. For use in computing derivatives of an optimization objective, 
    J(z(x), x), set v = dJ/dz,
        v = [dJ/diota, dJ/dsigma_1, ..., dJ/dsigma_nphi].

    Args:
        v (tensor): tensor of length nphi (number of residuals)
    
    Return:
        dsigma_by_ddofs (tuple): tuple of derivatives, one entry for each DOF. Each 
    """
    x = torch.clone(self.sigma)
    x[0] = self.iota

    # solve (1) for lambda (adjoint)
    dr_by_dsigma = self._jacobian(x)
    _lambda = torch.linalg.solve(dr_by_dsigma.T, - v)

    # use autodiff to compute transpose(dr/dx) * lambda
    r = self._residual(x)
    dofs = self.get_dofs(as_tuple=True)
    dsigma_iota_vjp_by_ddofs = torch.autograd.grad(r, dofs, grad_outputs=_lambda, retain_graph=True, allow_unused=True) # tuple
    
    return dsigma_iota_vjp_by_ddofs

def _determine_helicity(self):
    """
    Determine the integer N associated with the type of quasisymmetry
    by counting the number of times the normal vector rotates
    poloidally as you follow the axis around toroidally.
    """
    quadrant = torch.zeros(self.nphi + 1)
    for j in range(self.nphi):
        if self.normal_cylindrical[j,0] >= 0:
            if self.normal_cylindrical[j,2] >= 0:
                quadrant[j] = 1
            else:
                quadrant[j] = 4
        else:
            if self.normal_cylindrical[j,2] >= 0:
                quadrant[j] = 2
            else:
                quadrant[j] = 3
    quadrant[self.nphi] = quadrant[0]

    counter = 0
    for j in range(self.nphi):
        if quadrant[j] == 4 and quadrant[j+1] == 1:
            counter += 1
        elif quadrant[j] == 1 and quadrant[j+1] == 4:
            counter -= 1
        else:
            counter += quadrant[j+1] - quadrant[j]

    # It is necessary to flip the sign of axis_helicity in order
    # to maintain "iota_N = iota + axis_helicity" under the parity
    # transformations.
    counter *= self.spsi * self.sG
    self.helicity = counter / 4

def r1_diagnostics(self):
    """
    Compute various properties of the O(r^1) solution, once sigma and
    iota are solved for.
    """
    self.Y1s = self.sG * self.spsi * self.curvature / self.etabar
    self.Y1c = self.sG * self.spsi * self.curvature * self.sigma / self.etabar

    # If helicity is nonzero, then the original X1s/X1c/Y1s/Y1c variables are defined with respect to a "poloidal" angle that
    # is actually helical, with the theta=0 curve wrapping around the magnetic axis as you follow phi around toroidally. Therefore
    # here we convert to an untwisted poloidal angle, such that the theta=0 curve does not wrap around the axis.
    if self.helicity == 0:
        self.X1s_untwisted = self.X1s
        self.X1c_untwisted = self.X1c
        self.Y1s_untwisted = self.Y1s
        self.Y1c_untwisted = self.Y1c
    else:
        angle = -self.helicity * self.nfp * self.varphi
        sinangle = torch.sin(angle)
        cosangle = torch.cos(angle)
        self.X1s_untwisted = self.X1s *   cosangle  + self.X1c * sinangle
        self.X1c_untwisted = self.X1s * (-sinangle) + self.X1c * cosangle
        self.Y1s_untwisted = self.Y1s *   cosangle  + self.Y1c * sinangle
        self.Y1c_untwisted = self.Y1s * (-sinangle) + self.Y1c * cosangle

    # Use (R,Z) for elongation in the (R,Z) plane,
    # or use (X,Y) for elongation in the plane perpendicular to the magnetic axis.
    p = self.X1s * self.X1s + self.X1c * self.X1c + self.Y1s * self.Y1s + self.Y1c * self.Y1c
    q = self.X1s * self.Y1c - self.X1c * self.Y1s
    self.elongation = (p + torch.sqrt(p * p - 4 * q * q)) / (2 * torch.abs(q))
    self.mean_elongation = torch.sum(self.elongation * self.d_l_d_phi) / torch.sum(self.d_l_d_phi)
    index = torch.argmax(self.elongation)
    self.max_elongation = -fourier_minimum(-self.elongation.detach().numpy())

    self.d_X1c_d_varphi = torch.matmul(self.d_d_varphi, self.X1c)
    self.d_X1s_d_varphi = torch.matmul(self.d_d_varphi, self.X1s)
    self.d_Y1s_d_varphi = torch.matmul(self.d_d_varphi, self.Y1s)
    self.d_Y1c_d_varphi = torch.matmul(self.d_d_varphi, self.Y1c)

    self.calculate_grad_B_tensor()

