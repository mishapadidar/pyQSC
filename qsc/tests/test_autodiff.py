
import numpy as np
from qsc import Qsc
import torch

def finite_difference(f, x, eps=1e-6, *args, **kwargs):
    """Approximate jacobian with central difference.
    You should always clone and detach x before passing it to this function to prevent
    conflicts with aliasing and torch's autodiff graph.
        finite_difference(f, torch.clone(x).detach())

    Args:
        f (function): function to differentiate, can be scalar valued or 1d-array
            valued.
        x (tensor): input to f(x) at which to take the gradient.
        eps (float, optional): finite difference step size. Defaults to 1e-6.

    Returns:
        _type_: _description_
    """
    x = torch.clone(x)
    jac_est = []
    for i in range(len(x)):
        x[i] += eps
        fx = f(x, *args, **kwargs)
        x[i] -= 2*eps
        fy = f(x, *args, **kwargs)
        x[i] += eps
        jac_est.append((fx-fy)/(2*eps))
    jac = torch.stack(jac_est)
    if jac.ndim > 1:
        return jac.T
    else:
        return jac

def test_sigma_iota_derivatives():
    """
    Test the accuracy of the derivatives computed in the solve_state function: 
    derivatives of sigma and iota with respect to etabar, the axis shape coeffs.
    We check the derivatives with finite difference.
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", order='r1')

    # for name, param in stel.named_parameters():
    #  print(name, param)

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
    x = torch.clone(stel.rc.detach())
    deriv_fd = finite_difference(obj, x, 1e-6) # (nphi, dim_dof)
    vjp_fd = torch.matmul(deriv_fd.T, v)
    err = torch.max(torch.abs(vjp[0] - vjp_fd))
    print('dsigma/drc err', err.item())
    assert err < 1e-5

    # test derivative wrt etabar
    def obj(x):
        stel.etabar.data = x
        stel.calculate()
        sigma = torch.clone(stel.sigma).detach()
        sigma[0] = torch.clone(stel.iota).detach()
        return sigma
    x = torch.clone(torch.tensor([stel.etabar.detach()]))
    deriv_fd = finite_difference(obj, x, 1e-6).flatten() # (nphi, dim_dof)
    vjp_fd = torch.sum(deriv_fd * v)
    err = torch.max(torch.abs(vjp[4] - vjp_fd))
    print('dsigma/detabar err', err.item())
    assert err < 1e-5

def test_Bfield_axis_mse():
    """
    Test derivativative of Bfield_axis_mse with finite difference.
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", order='r1')

    # check loss
    stel.zero_grad()
    B_target = 1.4324 + torch.clone(stel.Bfield_cartesian()).detach()
    loss = stel.Bfield_axis_mse(B_target)

    # compute gradient 
    loss.backward()
    dloss_by_drc = stel.rc.grad

    # check gradient with finite difference
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        return stel.Bfield_axis_mse(B_target).detach()
    dloss_by_drc_fd = finite_difference(fd_obj, torch.clone(stel.rc.detach()), 1e-6)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print('dBfield_axis_mse/drc err', err.item())

def test_grad_B_tensor_cartesian_mse():
    """
    Test derivativative of grad_B_tensor_cartesian_mse with finite difference.
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", order='r1')

    # check loss
    gradB_target = 1.34 + torch.clone(stel.grad_B_tensor_cartesian()).detach()
    loss = stel.grad_B_tensor_cartesian_mse(gradB_target)
    print(loss)

    # compute gradient 
    stel.zero_grad()
    loss.backward(retain_graph=True)

    # stel.dresidual_by_ddof_vjp(torch.zeros(stel.nphi))

    # chain rule
    partialloss_by_partial_rc = stel.rc.grad 
    partialloss_by_partial_zs = stel.zs.grad 
    partialloss_by_partial_etabar = stel.etabar.grad
    partialloss_by_partialsigma = stel.sigma.grad 
    partialloss_by_partialiota = stel.iota.grad 
    partialloss_by_partialz = torch.clone(partialloss_by_partialsigma).detach()
    partialloss_by_partialz[0] = torch.clone(partialloss_by_partialiota).detach()
    partialloss_by_partial_dofs = stel.dresidual_by_ddof_vjp(partialloss_by_partialz)
    partialloss_by_partial_rc_indirect = partialloss_by_partial_dofs[0]
    partialloss_by_partial_zs_indirect = partialloss_by_partial_dofs[1]
    partialloss_by_partial_etabar_indirect = partialloss_by_partial_dofs[4]
    dloss_by_drc = partialloss_by_partial_rc_indirect + partialloss_by_partial_rc
    dloss_by_dzs = partialloss_by_partial_zs_indirect + partialloss_by_partial_zs
    dloss_by_detabar = partialloss_by_partial_etabar_indirect + partialloss_by_partial_etabar

    # check rc gradient with finite difference
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        return stel.grad_B_tensor_cartesian_mse(gradB_target).detach()
    dloss_by_drc_fd = finite_difference(fd_obj, torch.clone(stel.rc.detach()), 1e-7)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print('dgrad_B_tensor_cartesian_mse/drc err', err.item())

    # check zs gradient with finite difference
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        return stel.grad_B_tensor_cartesian_mse(gradB_target).detach()
    dloss_by_dzs_fd = finite_difference(fd_obj, torch.clone(stel.zs.detach()), 1e-7)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    print('dgrad_B_tensor_cartesian_mse/dzs err', err.item())

    # check etabar gradient with finite difference
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        return stel.grad_B_tensor_cartesian_mse(gradB_target).detach()
    dloss_by_detabar_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-7)
    err = torch.max(torch.abs(dloss_by_detabar - dloss_by_detabar_fd))
    print('dgrad_B_tensor_cartesian_mse/detabar err', err.item())


if __name__ == "__main__":
    test_sigma_iota_derivatives()
    test_Bfield_axis_mse()
    test_grad_B_tensor_cartesian_mse()