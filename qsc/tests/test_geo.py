
import numpy as np
from qsc.qsc import Qsc
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def test_surface():
    """
    Test the accuracy of the surface computation
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", order='r2')

    minor_radius = 0.1
    ntheta = 16
    X,Y,Z,R = stel.get_boundary(r=minor_radius, ntheta=ntheta, nphi = stel.nphi, ntheta_fourier=256, mpol=25, ntor=25) # (nphi, ntheta)
    xyz = stel.surface(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    dxyz_by_dvarphi = stel.dsurface_by_dvarphi(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    dxyz_by_dtheta = stel.dsurface_by_dtheta(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    normal = stel.surface_normal(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)

    # sanity check plot
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_wireframe(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], alpha=0.2) # ours
    ax.quiver(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], normal[:,:,0], normal[:,:,1], normal[:,:,2], alpha=0.2) # ours

    ax.plot_wireframe(X, Y, Z, alpha=0.2, color='orange') # qsc
    plt.show()

def test_surface_tangents():
    # set up the expansion
    stel = Qsc.from_paper("precise QA", nphi=501, order='r2')

    minor_radius = 0.1
    ntheta = 512
    xyz = stel.surface(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    dxyz_by_dvarphi = stel.dsurface_by_dvarphi(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    dxyz_by_dtheta = stel.dsurface_by_dtheta(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    dxyz_by_dr = stel.dsurface_by_dr(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    normal = stel.surface_normal(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    
    # test normal is orthogonal to tangent
    err = np.max(np.abs(np.sum(normal * dxyz_by_dtheta,axis=-1)))
    print(err)
    assert err < 1e-4, "normal not orthogonal to theta tangent"

    # test normal is orthogonal to tangent
    err = np.max(np.abs(np.sum(normal * dxyz_by_dvarphi,axis=-1)))
    print(err)
    assert err < 1e-4, "normal not orthogonal to varphi tangent"

    # test dsurface_by_dtheta with central differences
    dtheta = 2 * np.pi / ntheta
    dxyz_by_dtheta_fd = (xyz[:,2:,:] - xyz[:,:-2,:]) / (2 * dtheta)
    err = np.max(np.abs(dxyz_by_dtheta[:,1:-1,:] - dxyz_by_dtheta_fd))
    print(err)
    assert err < 1e-4, "dsurface_by_dtheta incorrect"

    
    # test dsurface_by_dr with central differences
    dr = 1e-2
    xyz_pdr = stel.surface(r=minor_radius + dr, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    xyz_mdr = stel.surface(r=minor_radius - dr, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    dxyz_by_dr_fd = (xyz_pdr - xyz_mdr) / (2 * dr)
    err = np.max(np.abs(dxyz_by_dr - dxyz_by_dr_fd))
    print(err)
    assert err < 1e-4, "dsurface_by_dr incorrect"
    
    # test dsurface_by_dvarphi with central differences
    dphi = 2 * np.pi / (stel.nfp * stel.nphi)
    dxyz_by_dphi_fd = (xyz[2:,:,:] - xyz[:-2,:,:]) / (2 * dphi)
    d_varphi_d_phi = stel.d_varphi_d_phi.detach().numpy()[1:-1].reshape((-1,1,1))
    dxyz_by_dvarphi_fd = dxyz_by_dphi_fd / d_varphi_d_phi
    err = np.max(np.abs(dxyz_by_dvarphi[1:-1,:,:] - dxyz_by_dvarphi_fd))
    print(err)
    assert err < 1e-4, "dsurface_by_dvarphi incorrect"

if __name__ == "__main__":
    # test_surface()
    test_surface_tangents()