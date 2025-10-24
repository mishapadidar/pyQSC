
import numpy as np
from qsc.qsc import Qsc
import torch
import matplotlib.pyplot as plt
from qsc.util import finite_difference_torch


def test_r1_diagnostics():

    names = ["precise QH", "precise QA"]
    for name in names:
        stel = Qsc.from_paper(name, I2 = 1.0, order='r1')
        stel_vac = Qsc.from_paper(name, I2 = 0.0, order='r1')

        # total vacuum solution and vacuum components should match
        assert torch.allclose(stel_vac.X1c, stel_vac.X1c_vac, atol=1e-14), "X1c or X1c_vac incorect in vacuum"
        assert torch.allclose(stel_vac.X1s, stel_vac.X1s_vac, atol=1e-14), "X1s or X1s_vac incorect in vacuum"
        assert torch.allclose(stel_vac.Y1s, stel_vac.Y1s_vac, atol=1e-14), "Y1s or Y1s_vac incorect in vacuum"
        assert torch.allclose(stel_vac.Y1c, stel_vac.Y1c_vac, atol=1e-14), "Y1c or Y1c_vac incorect in vacuum"
        assert torch.allclose(stel_vac.Y1c_untwisted, stel_vac.Y1c_vac_untwisted, atol=1e-14), "Y1c_untwisted or Y1c_vac_untwisted incorect in vacuum"
        assert torch.allclose(stel_vac.sigma, stel_vac.sigma_vac, atol=1e-14), "sigma or sigma_vac incorect in vacuum"
        assert torch.allclose(stel_vac.iota, stel_vac.iota_vac, atol=1e-14), "iota or iota_vac incorect in vacuum"

        # vacuum component of nonvac field should match the total vacuum solution
        assert torch.allclose(stel_vac.X1c, stel.X1c_vac, atol=1e-14), "X1c or X1c_vac incorect"
        assert torch.allclose(stel_vac.X1s, stel.X1s_vac, atol=1e-14), "X1s or X1s_vac incorect"
        assert torch.allclose(stel_vac.Y1s, stel.Y1s_vac, atol=1e-14), "Y1s or Y1s_vac incorect"
        assert torch.allclose(stel_vac.Y1c, stel.Y1c_vac, atol=1e-14), "Y1c or Y1c_vac incorect"
        assert torch.allclose(stel_vac.Y1c_untwisted, stel.Y1c_vac_untwisted, atol=1e-14), "Y1c_untwisted or Y1c_vac_untwisted incorect"
        assert torch.allclose(stel_vac.sigma, stel.sigma_vac, atol=1e-14), "sigma or sigma_vac incorect"
        assert torch.allclose(stel_vac.iota, stel.iota_vac, atol=1e-14), "iota or iota_vac incorect"

    print('PASSED: test_r1_diagnostics')

def test_dresidual_vac_by_ddof_vjp():
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
    vjp = stel.dresidual_vac_by_ddof_vjp(v) # tuple, one entry per dof tensor

    # test derivative wrt rc
    def obj(x):
        stel.rc.data = x
        stel.calculate()
        sigma = torch.clone(stel.sigma_vac).detach()
        sigma[0] = torch.clone(stel.iota_vac).detach()
        return sigma
    x = torch.clone(stel.rc.detach())
    deriv_fd = finite_difference_torch(obj, x, 1e-6) # (nphi, dim_dof)
    vjp_fd = torch.matmul(deriv_fd.T, v)
    err = torch.max(torch.abs(vjp[0] - vjp_fd))
    # print('dsigma_vac/drc err', err.item())
    assert err < 1e-5, "dsigma_iota_vac/drc failed"

    # test derivative wrt etabar
    def obj(x):
        stel.etabar.data = x
        stel.calculate()
        sigma = torch.clone(stel.sigma_vac).detach()
        sigma[0] = torch.clone(stel.iota_vac).detach()
        return sigma
    x = torch.clone(torch.tensor([stel.etabar.detach()]))
    deriv_fd = finite_difference_torch(obj, x, 1e-4).flatten() # (nphi, dim_dof)
    vjp_fd = torch.sum(deriv_fd * v)
    err = torch.max(torch.abs(vjp[4] - vjp_fd))
    # print('dsigma_vac/detabar err', err.item())
    assert err < 1e-5, "dsigma_iota_vac/detabar failed"

    print('PASSED: test_dresidual_vac_by_ddof_vjp')


if __name__ == "__main__":
    test_r1_diagnostics()
    test_dresidual_vac_by_ddof_vjp()
