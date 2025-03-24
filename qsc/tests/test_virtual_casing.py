
import numpy as np
from qsc.qsc import Qsc
import torch
import matplotlib.pyplot as plt
from qsc.util import finite_difference


def test_B_external_on_axis():
    """
    Test the accuracy of the surface computation
    """
    # # set up the expansion
    # stel = Qsc.from_paper("precise QA", nphi=2001, order='r2')

    # minor_radius = 0.1
    # ntheta = 256
    # n_target = 32
    # idx_target = range(0, stel.nphi, n_target)
    # X_target = stel.XYZ0[:,idx_target].T

    # # # TODO: remove
    # # X_target = [X_target[-1]]
    # # Bext = stel.Bfield_cartesian()[:,idx_target][:,-1:] # (3, nphi)

    # # single evaluation
    # with torch.no_grad():
    #     # Bext_vc = stel.B_external_on_axis(r=minor_radius, ntheta=ntheta, X_target = X_target) # (3, nphi)
    #     Bext_vc = stel.B_external_on_axis_taylor(r=minor_radius, ntheta=ntheta, X_target = X_target) # (3, nphi)
    # Bext = stel.Bfield_cartesian()[:,idx_target] # (3, nphi)

    # err = Bext - Bext_vc
    # # print(err)
    # print(torch.max(torch.abs(err)))

    """ plot error against minor radius"""
    mr_list = [0.01, 0.02, 0.04, 0.08, 0.1, 0.12, 0.14]
    ntheta = 256
    n_target = 32
    # storage 
    errs = []
    for mr in mr_list:
        stel = Qsc.from_paper("precise QA", nphi=2001, order='r2')
        idx_target = range(0, stel.nphi, n_target)
        X_target = stel.XYZ0[:,idx_target].T
        with torch.no_grad():
            # Bext_vc = stel.B_external_on_axis(r=mr, ntheta=ntheta, X_target = X_target) # (3, nphi)
            Bext_vc = stel.B_external_on_axis_taylor(r=mr, ntheta=ntheta, X_target = X_target) # (3, nphi)
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

def test_grad_B_external_on_axis():
    """
    Test the accuracy of the surface computation
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", nphi=511, order='r1')

    minor_radius = 0.2
    ntheta = 256
    idx_target = range(0, stel.nphi, 64)
    X_target = stel.XYZ0[:,idx_target].T

    # single evaluation
    with torch.no_grad():
        grad_Bext_vc = stel.grad_B_external_on_axis(r=minor_radius, ntheta=ntheta, X_target = X_target) # (3, nphi)
    grad_Bext_vc = grad_Bext_vc.detach().numpy()

    # compute the derivative with finite difference
    def fd_obj(X, ii):
        X = torch.tensor(X).reshape((1,-1))
        with torch.no_grad():
            Bext_vc = stel.B_external_on_axis(r=minor_radius, ntheta=ntheta, X_target = X) # (3, nphi)
        return Bext_vc.detach().numpy()
    
    grad_Bext_fd = np.zeros(np.shape(grad_Bext_vc))
    for ii, x in enumerate(X_target.detach().numpy()):
        grad_Bext_fd[:,:,ii] = finite_difference(fd_obj, x, 1e-3, ii=ii)[0]

    err = grad_Bext_vc - grad_Bext_fd
    print(np.max(np.abs(err)))
    assert np.max(np.abs(err)) < 1e-5, "grad_B_external is incorrect"


def test_n_cross_B():
    # set up the expansion
    stel1 = Qsc.from_paper("precise QA", nphi=511, order='r1')
    stel2 = Qsc.from_paper("precise QA", nphi=511, order='r1')

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


if __name__ == "__main__":
    # test_B_external_on_axis()
    test_n_cross_B()
    # test_grad_B_external_on_axis()