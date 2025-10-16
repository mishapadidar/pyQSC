
import numpy as np
from qsc.qsc import Qsc
import torch
import matplotlib.pyplot as plt
from qsc.util import finite_difference, finite_difference_torch


def test_B_external_on_axis_taylor():
    """
    Test the accuracy of the B_external computation
    """

    """ plot error against minor radius. Should be O(r^3)"""
    mr_list = [0.01, 0.02, 0.04, 0.08, 0.1, 0.12, 0.14]
    ntheta = 256
    nphi = 8192
    # storage 
    errs = []
    for mr in mr_list:
        stel = Qsc.from_paper("precise QA", nphi=61, order='r3')
        X_target = stel.XYZ0.T
        with torch.no_grad():
            Bext_vc = stel.B_external_on_axis_taylor(r=mr, ntheta=ntheta, nphi=nphi, X_target = X_target) # (3, nphi)

        Bext = stel.Bfield_cartesian() # (3, nphi)
        err = Bext - Bext_vc
        errs.append(torch.max(torch.abs(err)).detach().numpy())
    
    plt.plot(mr_list, errs, lw=3, marker='o', color='k')
    # plot a cubic e(r) = a*r^3
    a = errs[-1] / (mr_list[-1] **3)
    x = np.linspace(mr_list[0], mr_list[-1], 30)
    y = a * x**3
    plt.plot(x, y, lw=1, linestyle='--', color='k', label='$O(r^3)$')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('minor radius', fontsize=14)
    plt.ylabel('virtual casing error', fontsize=14)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def test_grad_B_external_on_axis_taylor_converges():
    """
    Show that the computation of grad_B_external_on_axis_taylor converges with nphi.
    We expect spectral convergence. 
    The error may grow with p2.
    """

    config = "2022 QH nfp3 beta"
    naxis = 61
    nphi_list = [256, 512, 1024, 2048]
    nphi_ref = 8192
    mr = 1/8
    ntheta = 256
    p2_list = [-2e1, -2e3, -2e5, -2e6]
    I2 = 0.0

    for p2 in p2_list:
        # storage
        errs = []

        stel = Qsc.from_paper(config, nphi=naxis, p2=p2, I2=I2)
        gradB_ref = stel.grad_B_external_on_axis_taylor(r=mr, ntheta=ntheta, nphi=nphi_ref)

        for nphi in nphi_list:
            with torch.no_grad():
                gradB_ext_vc = stel.grad_B_external_on_axis_taylor(r=mr, ntheta=ntheta, nphi=nphi)

            err = gradB_ref - gradB_ext_vc
            # errs.append(torch.max(torch.abs(err)).detach().numpy())
            rel_err = 100 * torch.linalg.norm(err, axis=(0,1)) / torch.linalg.norm(gradB_ref, axis=(0,1))
            errs.append(torch.max(rel_err).detach().numpy())
        plt.plot(nphi_list, errs, lw=3, marker='o', label=f'p2=%.2e'%p2)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('nphi', fontsize=14)
    plt.ylabel('relative error [%]', fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def test_grad_B_external_on_axis_taylor_accuracy():
    """
    Test the accuracy of the grad_B_external_on_axis_taylor computation in vacuum.
    Should converge at O(r^2).
    """

    """ plot error against minor radius"""
    mr_list = [0.01, 0.02, 0.04, 0.08, 0.1, 0.12, 0.14]
    ntheta = 256
    nphi = 8192
    n_axis = 61
    # storage 
    errs = []
    for mr in mr_list:
        stel = Qsc.from_paper("precise QA", nphi=n_axis, order='r2')
        X_target = stel.XYZ0.T
        with torch.no_grad():
            Bext_vc = stel.grad_B_external_on_axis_taylor(r=mr, ntheta=ntheta, nphi=nphi, X_target = X_target) # (3, 3, nphi)
        Bext = stel.grad_B_tensor_cartesian() # (3, 3, nphi)
        err = Bext - Bext_vc
        rel_err = 100 * torch.linalg.norm(err, axis=(0,1)) / torch.linalg.norm(Bext, axis=(0,1))
        errs.append(torch.max(rel_err).detach().numpy())

    plt.plot(mr_list, errs, lw=2, marker='o', color='k', label='error')

    # plot a quadratic e(r) = a*r^2
    a = errs[-1] / (mr_list[-1] **2)
    x = np.linspace(mr_list[0], mr_list[-1], 30)
    y = a * x**2
    plt.plot(x, y, lw=1, linestyle='--', color='k', label='$\mathcal{O}(r^2)$ reference')

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('minor radius', fontsize=14)
    plt.ylabel('Relative error [%]', fontsize=14)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def test_grad_B_external_on_axis_taylor_consistency():
    """
    Check that grad_B_external_on_axis_taylor is consistent with finite difference of B_external_on_axis_taylor.
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", nphi=61, order='r3', p2=0.0)

    minor_radius = 0.1
    ntheta = 256
    nphi = 2048

    X_target = torch.clone(stel.XYZ0.T)
    
    # evaluate gradient
    with torch.no_grad():
        grad_Bext_vc = stel.grad_B_external_on_axis_taylor(r=minor_radius, ntheta=ntheta, nphi=nphi, X_target = X_target) # (3, 3, nphi)
    grad_Bext_vc = grad_Bext_vc.detach().numpy()

    # function for finite difference
    def fd_obj(X):
        X = torch.tensor(X).reshape((1,-1))
        with torch.no_grad():
            Bext_vc = stel.B_external_on_axis_taylor(r=minor_radius, ntheta=ntheta, nphi=nphi, X_target = X) # (3, nphi)
        return Bext_vc.detach().numpy().flatten()
    
    # compute the derivative with finite difference
    grad_Bext_fd = np.zeros(np.shape(grad_Bext_vc))
    for ii, x in enumerate(X_target.detach().numpy()):
        grad_Bext_fd[:,:,ii] = finite_difference(fd_obj, x, 1e-4)

    err = grad_Bext_vc - grad_Bext_fd
    print(np.max(np.abs(err)))
    assert np.max(np.abs(err)) < 1e-5, "grad_B_external is incorrect"


def test_n_cross_B():
    """
    Compute the diffefence between two methods of computing n x B. The Taylor field method is a higher
    order method than the Boozer representation. We expect to see the predictions diverge as r^3
    as r approaches zero.
    """
    # set up the expansion
    stel1 = Qsc.from_paper("precise QA", nphi=511, order='r2')
    stel2 = Qsc.from_paper("precise QA", nphi=511, order='r2')

    r_list = torch.linspace(0.01, 1, 10)
    ntheta = 64

    error_array = np.zeros(len(r_list))
    for ii, r in enumerate(r_list):
        # compute n x B using Taylor B
        B_taylor = stel2.B_taylor(r=r, ntheta=ntheta)
        n = stel2.surface_normal(r=r, ntheta=ntheta)
        n_cross_B = torch.linalg.cross(n, B_taylor)

        # compute n x B using Boozer approach
        I = 0.0
        G = stel1.G0
        if stel1.order != 'r1':
            I += r**2 * stel1.I2
            G += r**2 * stel1.G2
        dr_by_dtheta = stel1.dsurface_by_dtheta(r=r, ntheta=ntheta) # (nphi, ntheta, 3)
        dr_by_dvarphi = stel1.dsurface_by_dvarphi(r=r, ntheta=ntheta) # (nphi, ntheta, 3)
        n_cross_B_boozer = I * dr_by_dvarphi - G * dr_by_dtheta

        error_array[ii] = torch.max(torch.abs(n_cross_B - n_cross_B_boozer)).detach().numpy()
    
    plt.plot(r_list, error_array, lw=2, label='data')
    c = error_array[0]/r_list[0]**3
    plt.plot(r_list, c*r_list**3, '--', color='k', lw=2, label='r^3')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.ylabel('error in $n x B$')
    plt.xlabel('$r$')
    plt.show()


def test_B_external_on_axis_taylor_autodiff():
    """
    Test autodifferentation of B_external_on_axis_taylor. 
    Compare against finite difference.
    """
    ntheta = 256
    nphi=2048
    r = 0.1
    
    stel = Qsc.from_paper("precise QA", nphi=61, I2=1.0, p2=-1e3, order='r2')
    
    # mean squared error
    B_ext = stel.B_external_on_axis_taylor(r=r, ntheta=ntheta, nphi=nphi) # (3, nphi)
    mean = torch.mean(B_ext.detach())
    loss = torch.mean((B_ext - mean)**2)

    # compute the gradient using autodiff
    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]
    dloss_by_dp2 = dloss_by_ddofs[7]
    dloss_by_dI2 = dloss_by_ddofs[8]

    # check rc gradient with finite difference
    x0 = torch.clone(stel.rc.detach())
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        B_ext = stel.B_external_on_axis_taylor(r=r, ntheta=ntheta, nphi=nphi) # (3, nphi)
        loss = torch.mean((B_ext - mean)**2).detach()
        return loss
    dloss_by_drc_fd = finite_difference_torch(fd_obj, x0, 1e-8)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print(err.item())
    assert err.item() < 1e-3, f"dloss/drc finite difference check failed: {err.item()}"
    stel.rc.data = x0 # restore the original value after finite difference check

    # check rc gradient with finite difference
    x0 = torch.clone(stel.zs.detach())
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        B_ext = stel.B_external_on_axis_taylor(r=r, ntheta=ntheta, nphi=nphi) # (3, nphi)
        loss = torch.mean((B_ext - mean)**2).detach()
        return loss
    dloss_by_dzs_fd = finite_difference_torch(fd_obj, x0, 1e-8)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    print(err.item())
    assert err.item() < 1e-3, f"dloss/dzs finite difference check failed: {err.item()}"
    stel.zs.data = x0 # restore the original value after finite difference check

    # check etabar gradient with finite difference
    x0 = torch.clone(torch.tensor([stel.etabar.detach()]))
    def fd_obj(x):
        stel.etabar.data = x[0]
        stel.calculate()
        B_ext = stel.B_external_on_axis_taylor(r=r, ntheta=ntheta, nphi=nphi) # (3, nphi)
        loss = torch.mean((B_ext - mean)**2).detach()
        return loss
    dloss_by_detabar_fd = finite_difference_torch(fd_obj, x0, 1e-4)
    err = torch.abs(dloss_by_detabar - dloss_by_detabar_fd)
    print(err.item())
    assert err.item() < 1e-3, f"dloss/detabar finite difference check failed: {err.item()}"
    stel.etabar.data = x0[0] # restore the original value after finite difference check

    # check p2 gradient with finite difference
    x0 = torch.clone(torch.tensor([stel.p2.detach()]))
    def fd_obj(x):
        stel.p2.data = x[0]
        stel.calculate()
        B_ext = stel.B_external_on_axis_taylor(r=r, ntheta=ntheta, nphi=nphi) # (3, nphi)
        loss = torch.mean((B_ext - mean)**2).detach()
        return loss
    dloss_by_dp2_fd = finite_difference_torch(fd_obj, x0, 1e-4)
    err = torch.abs(dloss_by_dp2 - dloss_by_dp2_fd)
    print(err.item())
    assert err.item() < 1e-3, f"dloss/dp2 finite difference check failed: {err.item()}"
    stel.p2.data = x0[0] # restore the original value after finite difference check

    # check I2 gradient with finite difference
    x0 = torch.clone(torch.tensor([stel.I2.detach()]))
    def fd_obj(x):
        stel.I2.data = x[0]
        stel.calculate()
        B_ext = stel.B_external_on_axis_taylor(r=r, ntheta=ntheta, nphi=nphi) # (3, nphi)
        loss = torch.mean((B_ext - mean)**2).detach()
        return loss
    dloss_by_dI2_fd = finite_difference_torch(fd_obj, x0, 1e-4)
    err = torch.abs(dloss_by_dI2 - dloss_by_dI2_fd)
    print(err.item())
    assert err.item() < 1e-3, f"dloss/dI2 finite difference check failed: {err.item()}"
    stel.I2.data = x0[0] # restore the original value after finite difference check


def test_B_taylor_autodiff():
    """
    Test autodifferentation of B_taylor
    """
    ntheta = 256
    r = 0.1
    vacuum_component=False
    
    stel = Qsc.from_paper("precise QA", nphi=61, I2=1.0, p2=-1e3, order='r2')

    B_taylor = stel.B_taylor(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (3, nphi)
    mean = torch.mean(B_taylor).detach()
    loss = torch.mean((B_taylor - mean)**2)

    # compute the gradient using autodiff
    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]
    dloss_by_dp2 = dloss_by_ddofs[7]

    # check rc gradient with finite difference
    x0 = torch.clone(stel.rc.detach())
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        B_taylor = stel.B_taylor(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (3, nphi)
        loss = torch.mean((B_taylor - mean)**2).detach()
        return loss
    dloss_by_drc_fd = finite_difference_torch(fd_obj, x0, 1e-9)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print(err.item())
    assert err.item() < 1e-3, f"dloss/drc finite difference check failed: {err.item()}"
    stel.rc.data = x0 # restore the original value after finite difference check

    # check zs gradient with finite difference
    x0 = torch.clone(stel.zs.detach())
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        B_taylor = stel.B_taylor(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (3, nphi)
        loss = torch.mean((B_taylor - mean)**2).detach()
        return loss
    dloss_by_dzs_fd = finite_difference_torch(fd_obj, x0, 1e-9)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    print(err.item())
    assert err.item() < 1e-3, f"dloss/dzs finite difference check failed: {err.item()}"
    stel.zs.data = x0 # restore the original value after finite difference check

    # check etabar gradient with finite difference
    x0 = torch.clone(torch.tensor([stel.etabar.detach()]))
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        B_taylor = stel.B_taylor(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (3, nphi)
        loss = torch.mean((B_taylor - mean)**2).detach()
        return loss
    dloss_by_detabar_fd = finite_difference_torch(fd_obj, x0, 1e-6)
    err = torch.abs(dloss_by_detabar - dloss_by_detabar_fd)
    print(err.item())
    assert err.item() < 1e-3, f"dloss/detabar finite difference check failed: {err.item()}"
    stel.etabar.data = x0 # restore the original value after finite difference check

    # check p2 gradient with finite difference
    x0 = torch.clone(torch.tensor([stel.p2.detach()]))
    def fd_obj(x):
        stel.p2.data = x
        stel.calculate()
        B_taylor = stel.B_taylor(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (3, nphi)
        loss = torch.mean((B_taylor - mean)**2).detach()
        return loss
    dloss_by_dp2_fd = finite_difference_torch(fd_obj, x0, 1e-3)
    err = torch.abs(dloss_by_dp2 - dloss_by_dp2_fd)
    print(err.item())
    assert err.item() < 1e-3, f"dloss/dp2 finite difference check failed: {err.item()}"
    stel.p2.data = x0 # restore the original value after finite difference check

def test_div_and_curl():
    """ Test that the divergence and curl are zero of the Taylor field are zero."""
    stel = Qsc.from_paper("precise QA", I2=0.0, p2=0.0, order='r2', nphi=301)

    r = 0.1
    ntheta = 121

    curl = stel.curl_taylor(r, ntheta=ntheta, vacuum_component=True) 
    err = torch.max(torch.norm(curl, dim=2)).detach().numpy() # (nphi, ntheta)
    print(err)
    assert err < 1e-10, f"curl is nonzero: {err.item()}"

    # divergence
    div = stel.divergence_taylor(r, ntheta=ntheta, vacuum_component=True)
    err = torch.max(torch.abs(div)).detach().numpy()
    print(err) # (nphi, ntheta)
    assert err < 1e-9, f"div is nonzero: {err.item()}"


if __name__ == "__main__":
    test_B_external_on_axis_taylor()
    test_grad_B_external_on_axis_taylor_converges()
    test_grad_B_external_on_axis_taylor_accuracy()
    test_grad_B_external_on_axis_taylor_consistency()
    test_n_cross_B()
    test_B_external_on_axis_taylor_autodiff()
    test_B_taylor_autodiff()
    test_div_and_curl()
