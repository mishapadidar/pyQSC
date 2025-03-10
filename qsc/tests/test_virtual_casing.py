
import numpy as np
from qsc.qsc import Qsc
import torch
import matplotlib.pyplot as plt


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

def test_n_cross_B():
    # set up the expansion
    stel1 = Qsc.from_paper("precise QA", nphi=511, order='r1')
    stel2 = Qsc.from_paper("precise QA", nphi=511, order='r2')

    r = 0.08
    ntheta = 64

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

    err = n_cross_B - n_cross_B_boozer
    print(torch.max(torch.abs(err)))

# def test_curl_B_taylor():
#     """ 
#     curl(B) = i(dyB3 - dzB2) - j(dxB3 - dzB1) + k(dxB2 - dyB1)
#     --------
#     jac(B) = dB/dr * grad(r) + dB/dtheta * grad(theta) + dB/dphi*grad(phi)
#     where * means outer product
#     """
#     stel = Qsc.from_paper("precise QA", nphi=511, order='r1')

#     r = 0.1
#     ntheta = 64
#     ds_by_dr = stel.dsurface_by_dr(r,ntheta)
#     ds_by_dtheta = stel.dsurface_by_dtheta(r,ntheta)
#     ds_by_dvarphi = stel.dsurface_by_dvarphi(r,ntheta)

#     # TODO: check the cross is done right and the formula
#     sqrtg = ds_by_dr * torch.linalg.cross(ds_by_dtheta, ds_by_dvarphi)
#     grad_r = torch.linalg.cross(ds_by_dtheta, ds_by_dvarphi) / sqrtg
#     grad_theta = torch.linalg.cross(ds_by_dvarphi, ds_by_dr) / sqrtg
#     grad_varphi = torch.linalg.cross(ds_by_dr, ds_by_dtheta) / sqrtg

#     # TODO: analyticaly compute dB/dr, dB/dtheta
#     dr = 1e-4
#     Brph = stel.B_taylor(r+dr, ntheta)
#     Brmh = stel.B_taylor(r-dr, ntheta)
#     dB_by_dr = (Brph - Brmh) / (2 * dr) # (nphi, ntheta, 3)
#     B = stel.B_taylor(r, ntheta) # (nphi, ntheta, 3)
#     dtheta = 2 * np.pi / ntheta
#     dB_by_dtheta = (B[:, 2:, :] - B[:, :-2, :]) / (2 * dtheta) # (nphi, ntheta - 2, 3)
#     dB_by_dphi = (B[2:,:,:] - B[:-2,:,:]) / (2 * stel.dphi) # (nphi-2, ntheta, 3)
#     dB_by_dvarphi = dB_by_dphi / stel.d_varphi_d_phi.reshape((-1, 1, 1)) # (nphi-2, ntheta, 3)

#     # trim arrays to same size
#     dB_by_dr = dB_by_dr[1:-1, 1:-1, :] # (nphi-2, ntheta-2, 3)
#     dB_by_dtheta = dB_by_dtheta[1:-1,:,:] # (nphi-2, ntheta-2, 3)
#     dB_by_dvarphi = dB_by_dvarphi[:,1:-1,:] # (nphi-2, ntheta-2, 3)
#     grad_r = grad_r[1:-1, 1:-1, :] # (nphi-2, ntheta-2, 3)
#     grad_theta = grad_theta[1:-1, 1:-1, :] # (nphi-2, ntheta-2, 3)
#     grad_varphi = grad_varphi[1:-1, 1:-1, :] # (nphi-2, ntheta-2, 3)

#     # TODO: check the outer product is done right
#     # compile jac(B)
#     jacB = (torch.outer(dB_by_dr, grad_r) +
#             torch.outer(dB_by_dtheta, grad_theta) +
#             torch.outer(dB_by_dvarphi, grad_varphi)
#             ) # (nphi, ntheta, nB, nx)

#     # compute curl
#     curl_B = torch.zero((stel.nphi, ntheta, 3))
#     curl_B[:,:,0] = jacB[:,:,2,1] - jacB[:,:,1,2]
#     curl_B[:,:,1] = - (jacB[:,:,2,0] - jacB[:,:,0,2])
#     curl_B[:,:,2] = jacB[:,:,1,0] - jacB[:,:,0,1]


if __name__ == "__main__":
    test_B_external_on_axis()
    test_n_cross_B()
