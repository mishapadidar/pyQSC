
import numpy as np
from qsc import Qsc
import torch

def finite_difference(f, x, eps=1e-6, *args, **kwargs):
    """Approximate jacobian with central difference.

    Args:
        f (function): function to differentiate, can be scalar valued or 1d-array
            valued.
        x (1d-array): input to f(x) at which to take the gradient.
        eps (float, optional): finite difference step size. Defaults to 1e-6.

    Returns:
        _type_: _description_
    """
    jac_est = []
    for i in range(len(x)):
        x[i] += eps
        fx = f(x, *args, **kwargs)
        x[i] -= 2*eps
        fy = f(x, *args, **kwargs)
        x[i] += eps
        jac_est.append((fx-fy)/(2*eps))
    return torch.stack(jac_est).T

def test_sigma_iota_derivatives():
    """
    Test the accuracy of the derivatives computed in the solve_state function: 
    derivatives of sigma and iota with respect to etabar, the axis shape coeffs,
    and phi.
    We check the derivatives with finite difference.
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", order='r1')

    # compute derivatives of sigma and iota wrt dofs
    v = torch.ones(stel.nphi)/stel.nphi
    vjp = stel.dresidual_by_ddof_vjp(v) # tuple, one entry per dof tensor

    # test derivative wrt rc
    def obj(x):
        stel.rc.data = x
        stel.calculate()
        sigma = torch.clone(stel.sigma).detach()
        sigma[0] = torch.clone(stel.iota).detach()
        return sigma
    x = stel.rc.detach()
    h = 1e-6
    deriv_fd = finite_difference(obj, x, h) # (nphi, dim_dof)
    vjp_fd = torch.matmul(deriv_fd.T, v)
    err = torch.max(torch.abs(vjp[0] - vjp_fd))
    print(err)
    assert err < 1e-5

    # test derivative wrt etabar
    def obj(x):
        stel.etabar.data = x
        stel.calculate()
        sigma = torch.clone(stel.sigma).detach()
        sigma[0] = torch.clone(stel.iota).detach()
        return sigma
    x = stel.etabar.detach()
    obj0 = obj(x)
    h = 1e-5
    x += h
    objx = obj(x)
    deriv_fd = (objx - obj0)/h
    vjp_fd = torch.sum(deriv_fd * v)
    err = torch.max(torch.abs(vjp[4] - vjp_fd))
    print(err)
    assert err < 1e-5



if __name__ == "__main__":
    test_sigma_iota_derivatives()