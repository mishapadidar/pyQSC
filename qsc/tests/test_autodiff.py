
import numpy as np
from qsc.qsc import Qsc
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
    stel = Qsc.from_paper("precise QA", order='r2', I2=-0.1)

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

    # test derivative wrt I2
    def obj(x):
        stel.I2.data = x
        stel.calculate()
        sigma = torch.clone(stel.sigma).detach()
        sigma[0] = torch.clone(stel.iota).detach()
        return sigma
    x = torch.clone(torch.tensor([stel.I2.detach()]))
    deriv_fd = finite_difference(obj, x, 1e-6).flatten() # (nphi, dim_dof)
    vjp_fd = torch.sum(deriv_fd * v)
    err = torch.max(torch.abs(vjp[8] - vjp_fd))
    print('dsigma/I2 err', err.item())
    assert err < 1e-5

def test_sigma_iota_vac_derivatives():
    """
    Test the accuracy of the derivatives computed in the vacuum part of the solve_state function: 
    derivatives of sigma_vac and iota_vac with respect to etabar, the axis shape coeffs.
    We check the derivatives with finite difference.
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", order='r2')

    # for name, param in stel.named_parameters():
    #  print(name, param)

    # compute derivatives of sigma and iota wrt dofs
    v = torch.ones(stel.nphi)/stel.nphi
    vjp = stel.dresidual_by_ddof_vjp(v) # tuple, one entry per dof tensor

    # test derivative wrt rc
    def obj(x):
        stel.rc.data = x
        stel.calculate()
        sigma = torch.clone(stel.sigma_vac).detach()
        sigma[0] = torch.clone(stel.iota_vac).detach()
        return sigma
    x = torch.clone(stel.rc.detach())
    deriv_fd = finite_difference(obj, x, 1e-6) # (nphi, dim_dof)
    vjp_fd = torch.matmul(deriv_fd.T, v)
    err = torch.max(torch.abs(vjp[0] - vjp_fd))
    print('dsigma_vac/drc err', err.item())
    assert err < 1e-5

    # test derivative wrt etabar
    def obj(x):
        stel.etabar.data = x
        stel.calculate()
        sigma = torch.clone(stel.sigma_vac).detach()
        sigma[0] = torch.clone(stel.iota_vac).detach()
        return sigma
    x = torch.clone(torch.tensor([stel.etabar.detach()]))
    deriv_fd = finite_difference(obj, x, 1e-6).flatten() # (nphi, dim_dof)
    vjp_fd = torch.sum(deriv_fd * v)
    err = torch.max(torch.abs(vjp[4] - vjp_fd))
    print('dsigma_vac/detabar err', err.item())
    assert err < 1e-5

def test_Bfield_axis_mse():
    """
    Test derivativative of Bfield_axis_mse with finite difference.
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", order='r2')

    # check loss
    stel.zero_grad()
    B_target = 1.4324 + torch.clone(stel.Bfield_cartesian()).detach()
    loss = stel.Bfield_axis_mse(B_target)

    # compute gradient 
    # loss.backward()
    # dloss_by_drc = stel.rc.grad
    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_drs = dloss_by_ddofs[2]
    dloss_by_detabar = dloss_by_ddofs[4]

    # check gradient with finite difference
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        return stel.Bfield_axis_mse(B_target).detach()
    dloss_by_drc_fd = finite_difference(fd_obj, torch.clone(stel.rc.detach()), 1e-6)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print('dBfield_axis_mse/drc err', err.item())

    # check etabar gradient with finite difference
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        return stel.Bfield_axis_mse(B_target).detach()
    dloss_by_detabar_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-7)
    err = torch.max(torch.abs(dloss_by_detabar - dloss_by_detabar_fd))
    print('dBfield_axis_mse/detabar err', err.item())


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

    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]

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

def test_sum_loss():
    """
    Test gradient of a sum of losses with finite difference.
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", order='r1')

    # check loss
    gradB_target = 1.34 + torch.clone(stel.grad_B_tensor_cartesian()).detach()
    loss = stel.grad_B_tensor_cartesian_mse(gradB_target)
    B_target = 1.4324 + torch.clone(stel.Bfield_cartesian()).detach()
    loss += stel.Bfield_axis_mse(B_target)
    loss += (stel.iota - 1.21)**2
    print(loss)

    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]

    # check rc gradient with finite difference
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        l = stel.grad_B_tensor_cartesian_mse(gradB_target).detach()
        l += stel.Bfield_axis_mse(B_target).detach()
        l += (stel.iota.detach() - 1.21)**2
        return l
    dloss_by_drc_fd = finite_difference(fd_obj, torch.clone(stel.rc.detach()), 1e-7)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print('dloss/drc err', err.item())

    # check zs gradient with finite difference
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        l = stel.grad_B_tensor_cartesian_mse(gradB_target).detach()
        l += stel.Bfield_axis_mse(B_target).detach()
        l += (stel.iota.detach() - 1.21)**2
        return l
    dloss_by_dzs_fd = finite_difference(fd_obj, torch.clone(stel.zs.detach()), 1e-7)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    print('dloss/dzs err', err.item())

    # check etabar gradient with finite difference
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        l = stel.grad_B_tensor_cartesian_mse(gradB_target).detach()
        l += stel.Bfield_axis_mse(B_target).detach()
        l += (stel.iota.detach() - 1.21)**2
        return l
    dloss_by_detabar_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-7)
    err = torch.max(torch.abs(dloss_by_detabar - dloss_by_detabar_fd))
    print('dloss/detabar err', err.item())


def test_r2_derivatives():
    """
    Test the derivatives of the r2 quantities.
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", p2 = -1e5, I2 = -0.1, order='r2')
    rc0 = torch.clone(stel.rc).detach()
    zs0 = torch.clone(stel.zs).detach()
    etabar0 = torch.clone(stel.etabar).detach()
    p20 = torch.clone(stel.p2).detach()
    I20 = torch.clone(stel.I2).detach()

    """ Derivatives of X20 """
    loss = torch.mean(stel.X20**2)
    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]
    dloss_by_dp2 = dloss_by_ddofs[7]
    dloss_by_dI2 = dloss_by_ddofs[8]

    # check rc gradient with finite difference
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        return torch.mean(stel.X20**2).detach()
    dloss_by_drc_fd = finite_difference(fd_obj, torch.clone(stel.rc.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print('dX20/drc err', err.item())
    assert err.item() < 1e-4, "dX20/drc incorrect"

    # check zs gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        return torch.mean(stel.X20**2).detach()
    dloss_by_dzs_fd = finite_difference(fd_obj, torch.clone(stel.zs.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    print('dX20/dzs err', err.item())
    assert err.item() < 1e-4, "dX20/dzs incorrect"

    # check etabar gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        return torch.mean(stel.X20**2).detach()
    dloss_by_detabar_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_detabar - dloss_by_detabar_fd))
    print('dX20/detabar err', err.item())
    assert err.item() < 1e-4, "dX20/detabar incorrect"

    # check p2 gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.p2.data = p20
    stel.calculate()
    def fd_obj(x):
        stel.p2.data = x
        stel.calculate()
        return torch.mean(stel.X20**2).detach()
    dloss_by_dp2_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.p2.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_dp2 - dloss_by_dp2_fd))
    print('dX20/dp2 err', err.item())
    assert err.item() < 1e-4, "dX20/dp2 incorrect"

    # check I2 gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.p2.data = p20
    stel.I2.data = I20
    stel.calculate()
    def fd_obj(x):
        stel.I2.data = x
        stel.calculate()
        return torch.mean(stel.X20**2).detach()
    dloss_by_dI2_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.I2.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_dI2 - dloss_by_dI2_fd))
    print('dX20/dI2 err', err.item())
    assert err.item() < 1e-4, "dX20/dI2 incorrect"

    """ Derivatives of X2c """
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.p2.data = p20
    stel.I2.data = I20
    stel.calculate()
    loss = torch.mean(stel.X2c**2)
    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]
    dloss_by_dp2 = dloss_by_ddofs[7]
    dloss_by_dI2 = dloss_by_ddofs[8]

    # check rc gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        return torch.mean(stel.X2c**2).detach()
    dloss_by_drc_fd = finite_difference(fd_obj, torch.clone(stel.rc.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print('dX2c/drc err', err.item())
    assert err.item() < 1e-4, "dX2c/drc incorrect"

    # check rc gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        return torch.mean(stel.X2c**2).detach()
    dloss_by_dzs_fd = finite_difference(fd_obj, torch.clone(stel.zs.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    print('dX2c/dzs err', err.item())
    assert err.item() < 1e-4, "dX2c/dzs incorrect"

    # check etabar gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        return torch.mean(stel.X2c**2).detach()
    dloss_by_detabar_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_detabar - dloss_by_detabar_fd))
    print('dX2c/detabar err', err.item())
    assert err.item() < 1e-4, "dX2c/detabar incorrect"

    # check p2 gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.p2.data = p20
    stel.calculate()
    def fd_obj(x):
        stel.p2.data = x
        stel.calculate()
        return torch.mean(stel.X2c**2).detach()
    dloss_by_dp2_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.p2.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_dp2 - dloss_by_dp2_fd))
    print('X2c/dp2 err', err.item())
    assert err.item() < 1e-4, "X2c/dp2 incorrect"

    # check I2 gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.p2.data = p20
    stel.I2.data = I20
    stel.calculate()
    def fd_obj(x):
        stel.I2.data = x
        stel.calculate()
        return torch.mean(stel.X2c**2).detach()
    dloss_by_dI2_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.I2.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_dI2 - dloss_by_dI2_fd))
    print('dX2c/dI2 err', err.item())
    assert err.item() < 1e-4, "dX2c/dI2 incorrect"

    """ Derivatives of Y2s """
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.p2.data = p20
    stel.I2.data = I20
    stel.calculate()
    loss = torch.mean(stel.Y2s**2)
    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]
    dloss_by_dp2 = dloss_by_ddofs[7]
    dloss_by_dI2 = dloss_by_ddofs[8]

    # check rc gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        return torch.mean(stel.Y2s**2).detach()
    dloss_by_drc_fd = finite_difference(fd_obj, torch.clone(stel.rc.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print('dY2s/drc err', err.item())
    assert err.item() < 1e-4, "dY2s/drc incorrect"

    # check rc gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        return torch.mean(stel.Y2s**2).detach()
    dloss_by_dzs_fd = finite_difference(fd_obj, torch.clone(stel.zs.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    print('dY2s/dzs err', err.item())
    assert err.item() < 1e-4, "dY2s/dzs incorrect"

    # check etabar gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        return torch.mean(stel.Y2s**2).detach()
    dloss_by_detabar_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_detabar - dloss_by_detabar_fd))
    print('dY2s/detabar err', err.item())
    assert err.item() < 1e-4, "dY2s/detabar incorrect"

    # check p2 gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.p2.data = p20
    stel.calculate()
    def fd_obj(x):
        stel.p2.data = x
        stel.calculate()
        return torch.mean(stel.Y2s**2).detach()
    dloss_by_dp2_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.p2.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_dp2 - dloss_by_dp2_fd))
    print('dY2s/dp2 err', err.item())
    assert err.item() < 1e-4, "dY2s/dp2 incorrect"

    # check I2 gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.p2.data = p20
    stel.I2.data = I20
    stel.calculate()
    def fd_obj(x):
        stel.I2.data = x
        stel.calculate()
        return torch.mean(stel.Y2s**2).detach()
    dloss_by_dI2_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.I2.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_dI2 - dloss_by_dI2_fd))
    print('dY2s/dI2 err', err.item())
    assert err.item() < 1e-4, "dY2s/dI2 incorrect"

    """ Derivatives of G2 """
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.p2.data = p20
    stel.I2.data = I20
    stel.calculate()
    loss = torch.mean(stel.G2**2)
    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]
    dloss_by_dp2 = dloss_by_ddofs[7]
    dloss_by_dI2 = dloss_by_ddofs[8]

    # check rc gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        return torch.mean(stel.G2**2).detach()
    dloss_by_drc_fd = finite_difference(fd_obj, torch.clone(stel.rc.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print('dG2/drc err', err.item())
    assert err.item() < 1e-4, "dG2/drc incorrect"

    # check rc gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        return torch.mean(stel.G2**2).detach()
    dloss_by_dzs_fd = finite_difference(fd_obj, torch.clone(stel.zs.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    print('dG2/dzs err', err.item())
    assert err.item() < 1e-4, "dG2/dzs incorrect"

    # check etabar gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        return torch.mean(stel.G2**2).detach()
    dloss_by_detabar_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_detabar - dloss_by_detabar_fd))
    print('dG2/detabar err', err.item())
    assert err.item() < 1e-4, "dG2/detabar incorrect"

    # check p2 gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.p2.data = p20
    stel.calculate()
    def fd_obj(x):
        stel.p2.data = x
        stel.calculate()
        return torch.mean(stel.G2**2).detach()
    dloss_by_dp2_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.p2.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_dp2 - dloss_by_dp2_fd))
    print('dG2/dp2 err', err.item())
    assert err.item() < 1e-4, "dG2/dp2 incorrect"

    # check I2 gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.p2.data = p20
    stel.I2.data = I20
    stel.calculate()
    def fd_obj(x):
        stel.I2.data = x
        stel.calculate()
        return torch.mean(stel.G2**2).detach()
    dloss_by_dI2_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.I2.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_dI2 - dloss_by_dI2_fd))
    print('dG2/dI2 err', err.item())
    assert err.item() < 1e-4, "dG2/dI2 incorrect"

def test_r2_vac_derivatives():
    """
    Test the derivatives of the vacuum r2 quantities.
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", order='r2')
    rc0 = torch.clone(stel.rc).detach()
    zs0 = torch.clone(stel.zs).detach()
    etabar0 = torch.clone(stel.etabar).detach()

    """ Derivatives of X20 """
    loss = torch.mean(stel.X20_vac**2)
    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]

    # check rc gradient with finite difference
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        return torch.mean(stel.X20_vac**2).detach()
    dloss_by_drc_fd = finite_difference(fd_obj, torch.clone(stel.rc.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print('dX20_vac/drc err', err.item())
    assert err.item() < 1e-4, "dX20_vac/drc incorrect"

    # check zs gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        return torch.mean(stel.X20_vac**2).detach()
    dloss_by_dzs_fd = finite_difference(fd_obj, torch.clone(stel.zs.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    print('dX20_vac/dzs err', err.item())
    assert err.item() < 1e-4, "dX20_vac/dzs incorrect"

    # check etabar gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        return torch.mean(stel.X20_vac**2).detach()
    dloss_by_detabar_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_detabar - dloss_by_detabar_fd))
    print('dX20_vac/detabar err', err.item())
    assert err.item() < 1e-4, "dX20_vac/detabar incorrect"

    """ Derivatives of X2c """
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    loss = torch.mean(stel.X2c_vac**2)
    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]

    # check rc gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        return torch.mean(stel.X2c_vac**2).detach()
    dloss_by_drc_fd = finite_difference(fd_obj, torch.clone(stel.rc.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print('dX2c_vac/drc err', err.item())
    assert err.item() < 1e-4, "dX2c_vac/drc incorrect"

    # check rc gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        return torch.mean(stel.X2c_vac**2).detach()
    dloss_by_dzs_fd = finite_difference(fd_obj, torch.clone(stel.zs.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    print('dX2c_vac/dzs err', err.item())
    assert err.item() < 1e-4, "dX2c_vac/dzs incorrect"

    # check etabar gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        return torch.mean(stel.X2c_vac**2).detach()
    dloss_by_detabar_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_detabar - dloss_by_detabar_fd))
    print('dX2c_vac/detabar err', err.item())
    assert err.item() < 1e-4, "dX2c_vac/detabar incorrect"

    """ Derivatives of Y2s """
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    loss = torch.mean(stel.Y2s_vac**2)
    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]

    # check rc gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        return torch.mean(stel.Y2s_vac**2).detach()
    dloss_by_drc_fd = finite_difference(fd_obj, torch.clone(stel.rc.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print('dY2s_vac/drc err', err.item())
    assert err.item() < 1e-4, "dY2s_vac/drc incorrect"

    # check rc gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        return torch.mean(stel.Y2s_vac**2).detach()
    dloss_by_dzs_fd = finite_difference(fd_obj, torch.clone(stel.zs.detach()), 1e-9)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    print('dY2s_vac/dzs err', err.item())
    assert err.item() < 1e-4, "dY2s_vac/dzs incorrect"

    # check etabar gradient with finite difference
    stel.rc.data = rc0
    stel.zs.data = zs0
    stel.etabar.data = etabar0
    stel.calculate()
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        return torch.mean(stel.Y2s_vac**2).detach()
    dloss_by_detabar_fd = finite_difference(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-6)
    err = torch.max(torch.abs(dloss_by_detabar - dloss_by_detabar_fd))
    print('dY2s_vac/detabar err', err.item())
    assert err.item() < 1e-4, "dY2s_vac/detabar incorrect"

if __name__ == "__main__":
    test_sigma_iota_derivatives()
    test_sigma_iota_vac_derivatives()
    test_Bfield_axis_mse()
    test_grad_B_tensor_cartesian_mse()
    test_sum_loss()
    test_r2_derivatives()
    test_r2_vac_derivatives()