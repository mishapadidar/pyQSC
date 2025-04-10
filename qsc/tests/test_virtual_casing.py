
import numpy as np
from qsc.qsc import Qsc
import torch
import matplotlib.pyplot as plt
from qsc.util import finite_difference, finite_difference_torch


def test_B_external_on_axis():
    """
    Test the accuracy of the B_external computation
    """

    """ plot error against minor radius"""
    mr_list = [0.01, 0.02, 0.04, 0.08, 0.1, 0.12, 0.14]
    ntheta = 256
    nphi = 4096
    n_target = 32
    # storage 
    errs = []
    for mr in mr_list:
        stel = Qsc.from_paper("precise QA", nphi=61, order='r2')
        idx_target = range(0, stel.nphi, n_target)
        X_target = stel.XYZ0[:,idx_target].T
        with torch.no_grad():
            Bext_vc = stel.B_external_on_axis(r=mr, ntheta=ntheta, nphi=nphi, X_target = X_target) # (3, nphi)
            # Bext_vc = stel.B_external_on_axis_taylor(r=mr, ntheta=ntheta, nphi=nphi, X_target = X_target) # (3, nphi)
        Bext = stel.Bfield_cartesian()[:,idx_target] # (3, nphi)
        err = Bext - Bext_vc
        errs.append(torch.max(torch.abs(err)).detach().numpy())
    plt.plot(mr_list, errs, lw=3, marker='o', label='r2')

    # plot a quadratic e(r) = a*r^2
    a = errs[-1] / (mr_list[-1] **2)
    x = np.linspace(mr_list[0], mr_list[-1], 30)
    y = a * x**2
    plt.plot(x, y, lw=3, linestyle='--', label='theoretical')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('minor radius', fontsize=14)
    plt.ylabel('virtual casing error', fontsize=14)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def test_grad_B_external_on_axis_accuracy():
    """
    Test the accuracy of the surface computation
    """

    """ plot error against minor radius"""
    mr_list = [0.01, 0.02, 0.04, 0.08, 0.1, 0.12, 0.14]
    ntheta = 256
    nphi = 8192
    n_target = 32
    # storage 
    errs = []
    for mr in mr_list:
        stel = Qsc.from_paper("precise QA", nphi=31, order='r2')
        X_target = stel.XYZ0.T
        with torch.no_grad():
            Bext_vc = stel.grad_B_external_on_axis(r=mr, ntheta=ntheta, nphi=nphi, X_target = X_target) # (3, nphi)
            # Bext_vc = stel.B_external_on_axis_taylor(r=mr, ntheta=ntheta, nphi=nphi, X_target = X_target) # (3, nphi)
        Bext = stel.grad_B_tensor_cartesian() # (3, nphi)
        err = Bext - Bext_vc
        errs.append(torch.max(torch.abs(err)).detach().numpy())
    plt.plot(mr_list, errs, lw=3, marker='o', label='error')

    # plot a quadratic e(r) = a*r^2
    a = errs[-1] / (mr_list[-1] **2)
    x = np.linspace(mr_list[0], mr_list[-1], 30)
    y = a * x**2
    plt.plot(x, y, lw=3, linestyle='--', color='k', label='r^2 reference')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('minor radius', fontsize=14)
    plt.ylabel('virtual casing error', fontsize=14)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def test_grad_B_external_on_axis_consistency():
    """
    Test the accuracy of the surface computation
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", nphi=32, order='r1')

    minor_radius = 0.2
    ntheta = 256
    nphi = 2048
    idx_target = range(0, stel.nphi, 64)
    X_target = stel.XYZ0[:,idx_target].T

    # single evaluation
    with torch.no_grad():
        grad_Bext_vc = stel.grad_B_external_on_axis(r=minor_radius, ntheta=ntheta, nphi=nphi, X_target = X_target) # (3, nphi)
    grad_Bext_vc = grad_Bext_vc.detach().numpy()

    # compute the derivative with finite difference
    def fd_obj(X, ii):
        X = torch.tensor(X).reshape((1,-1))
        with torch.no_grad():
            Bext_vc = stel.B_external_on_axis(r=minor_radius, ntheta=ntheta, nphi=nphi, X_target = X) # (3, nphi)
        return Bext_vc.detach().numpy()
    
    grad_Bext_fd = np.zeros(np.shape(grad_Bext_vc))
    for ii, x in enumerate(X_target.detach().numpy()):
        grad_Bext_fd[:,:,ii] = finite_difference(fd_obj, x, 1e-4, ii=ii)[0]

    err = grad_Bext_vc - grad_Bext_fd
    print(np.max(np.abs(err)))
    assert np.max(np.abs(err)) < 1e-5, "grad_B_external is incorrect"


def test_n_cross_B():
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
    c = error_array[0]/r_list[0]**2
    plt.plot(r_list, c*r_list**2, '--', color='grey', lw=2, label='r^2')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.ylabel('error in $n x B$')
    plt.xlabel('$r$')
    plt.show()

def test_B_external_on_axis_split():
    """
    Test the accuracy of the B_external_on_axis_split computation
    """
    mr_list = [0.01, 0.02, 0.04, 0.08, 0.1, 0.12, 0.14]
    ntheta = 256
    nphi = 4096

    """ Check accuracy in vacuum """
    stel = Qsc.from_paper("precise QA", nphi=31, I2=0.0, p2=0.0, order='r2')
    Bext = stel.Bfield_cartesian()# (3, nphi)

    # storage 
    errs = []
    for mr in mr_list:
        with torch.no_grad():
            Bext_vc = stel.B_external_on_axis_split(r=mr, ntheta=ntheta, nphi=nphi) # (3, nphi)
        err = Bext - Bext_vc
        errs.append(torch.max(torch.abs(err)).detach().numpy())
    
    # should be zero error!
    assert np.allclose(errs, 0.0), "B_external_on_axis_split incorrect in vacuum case"

    """ Error of original calculation vs minor radius in vacuum """

    stel = Qsc.from_paper("precise QA", nphi=61, order='r2', p2 = 0.0, I2=0.0)
    Bext = stel.Bfield_cartesian()# (3, nphi)
    errs_taylor = []
    errs_split = []
    for mr in mr_list:
        with torch.no_grad():
            Bext_taylor = stel.B_external_on_axis_taylor(r=mr, ntheta=ntheta, nphi=nphi) # (3, nphi)
            Bext_split = stel.B_external_on_axis_split(r=mr, ntheta=ntheta, nphi=nphi) # (3, nphi)
        err = Bext - Bext_taylor
        errs_taylor.append(torch.max(torch.abs(err)).detach().numpy())
        err = Bext - Bext_split
        errs_split.append(torch.max(torch.abs(err)).detach().numpy())
    
    major_radius = float(stel.rc[0].detach().numpy().item())
    aspect_ratio = major_radius / np.array(mr_list)
    # taylor method errors
    plt.plot(aspect_ratio, errs_taylor, lw=3, color='tab:blue', marker='o', label='Naive Method')
    plt.plot(aspect_ratio, errs_split, lw=3, color='tab:orange', marker='o', label='Splitting Method')
    plt.grid(color='grey', linewidth=1, alpha=0.7)
    plt.yscale('symlog', linthresh=1e-5)
    plt.xscale('log')
    plt.xlabel('Aspect Ratio', fontsize=14)
    plt.ylabel('$|B_{ext} - B_{ext}^{NAE}|_{\infty}$', fontsize=14)
    plt.title("Approximation Accuracy of $B_{ext}$ in Vacuum")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def test_grad_B_external_on_axis_split():
    """
    Test the accuracy of the grad_B_external_on_axis_split computation
    """
    mr_list = [0.01, 0.02, 0.04, 0.08, 0.1, 0.12, 0.14]
    ntheta = 256
    nphi = 4096

    """ Check accuracy in vacuum """
    stel = Qsc.from_paper("precise QA", nphi=31, I2=0.0, p2=0.0, order='r2')
    grad_B_vac = stel.grad_B_tensor_cartesian() # (3, 3, nphi)

    # storage 
    errs = []
    for mr in mr_list:
        with torch.no_grad():
            grad_Bext_vc = stel.grad_B_external_on_axis_split(r=mr, ntheta=ntheta, nphi=nphi) # (3, nphi)
        err = grad_B_vac - grad_Bext_vc
        errs.append(torch.max(torch.abs(err)).detach().numpy())
    
    # should be zero error!
    assert np.allclose(errs, 0.0), "grad_B_external_on_axis_split incorrect in vacuum case"

def test_B_external_on_axis_split_autodiff():
    """
    Test autodifferentation of B_external_on_axis_split
    """
    ntheta = 256
    nphi=2048
    r = 0.1
    
    stel = Qsc.from_paper("precise QA", nphi=61, I2=1.0, p2=-1e3, order='r2')
    
    # mean squared error
    B_ext = stel.B_external_on_axis_split(r=r, ntheta=ntheta, nphi=nphi) # (3, nphi)
    mean = torch.mean(B_ext).detach()
    loss = torch.mean((B_ext - mean)**2)

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
        B_ext = stel.B_external_on_axis_split(r=r, ntheta=ntheta, nphi=nphi) # (3, nphi)
        loss = torch.mean((B_ext - mean)**2).detach()
        return loss
    dloss_by_drc_fd = finite_difference_torch(fd_obj, x0, 1e-7)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    print(err.item())
    assert err.item() < 1e-3, f"dloss/drc finite difference check failed: {err.item()}"
    stel.rc.data = x0 # restore the original value after finite difference check

    # check rc gradient with finite difference
    x0 = torch.clone(stel.zs.detach())
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        B_ext = stel.B_external_on_axis_split(r=r, ntheta=ntheta, nphi=nphi) # (3, nphi)
        loss = torch.mean((B_ext - mean)**2).detach()
        return loss
    dloss_by_dzs_fd = finite_difference_torch(fd_obj, x0, 1e-7)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    print(err.item())
    assert err.item() < 1e-3, f"dloss/dzs finite difference check failed: {err.item()}"
    stel.zs.data = x0 # restore the original value after finite difference check

    # check etabar gradient with finite difference
    x0 = torch.clone(torch.tensor([stel.etabar.detach()]))
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        B_ext = stel.B_external_on_axis_split(r=r, ntheta=ntheta, nphi=nphi) # (3, nphi)
        loss = torch.mean((B_ext - mean)**2).detach()
        return loss
    dloss_by_detabar_fd = finite_difference_torch(fd_obj, x0, 1e-3)
    err = torch.abs(dloss_by_detabar - dloss_by_detabar_fd)
    print(err.item())
    assert err.item() < 1e-3, f"dloss/detabar finite difference check failed: {err.item()}"
    stel.etabar.data = x0 # restore the original value after finite difference check

    # check p2 gradient with finite difference
    x0 = torch.clone(torch.tensor([stel.p2.detach()]))
    def fd_obj(x):
        stel.p2.data = x
        stel.calculate()
        B_ext = stel.B_external_on_axis_split(r=r, ntheta=ntheta, nphi=nphi) # (3, nphi)
        loss = torch.mean((B_ext - mean)**2).detach()
        return loss
    dloss_by_dp2_fd = finite_difference_torch(fd_obj, x0, 1e-3)
    err = torch.abs(dloss_by_dp2 - dloss_by_dp2_fd)
    print(err.item())
    assert err.item() < 1e-3, f"dloss/dp2 finite difference check failed: {err.item()}"
    stel.p2.data = x0 # restore the original value after finite difference check

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

if __name__ == "__main__":
    test_B_external_on_axis()
    test_n_cross_B()
    test_grad_B_external_on_axis_accuracy()
    test_grad_B_external_on_axis_consistency()
    test_B_external_on_axis_split()
    test_grad_B_external_on_axis_split()
    test_B_external_on_axis_split_autodiff()
    test_B_taylor_autodiff()