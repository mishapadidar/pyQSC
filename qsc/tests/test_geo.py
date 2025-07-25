
import numpy as np
from qsc.qsc import Qsc
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from qsc.util import finite_difference_torch
from unittest.mock import patch



def test_surface():
    """
    Test the accuracy of the surface computation
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", order='r3')

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
    # ax.quiver(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], normal[:,:,0], normal[:,:,1], normal[:,:,2], alpha=0.2) # ours

    ax.plot_wireframe(X, Y, Z, alpha=0.2, color='orange') # qsc
    plt.show()

def test_surface_tangents():
    # set up the expansion
    stel = Qsc.from_paper("precise QA", nphi=513, order='r3')

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
    dr = 1e-3
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

def test_second_derivative():
    """ Test the accuracy of the second derivative of the surface."""
    # set up the expansion
    stel = Qsc.from_paper("precise QA", nphi=513, order='r3')

    minor_radius = 0.1
    ntheta = 512
    dxyz_by_dtheta = stel.dsurface_by_dtheta(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    d2xyz_by_dthetatheta = stel.d2surface_by_dthetatheta(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)

    # test dsurface_by_dtheta with central differences
    dtheta = 2 * np.pi / ntheta
    dxyz_by_dtheta_fd = (dxyz_by_dtheta[:,2:,:] - dxyz_by_dtheta[:,:-2,:]) / (2 * dtheta)
    err = np.max(np.abs(d2xyz_by_dthetatheta[:,1:-1,:] - dxyz_by_dtheta_fd))
    print(err)
    assert err < 1e-4, "d2surface_by_dthetatheta incorrect"

def test_surface_nonvac():
    """
    Test the accuracy of the nonvacuum surface computation
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", I2 = 10, order='r1')

    minor_radius = 0.1
    ntheta = 16
    surface = stel.surface(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    surface_nonvac = stel.surface_nonvac(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    surface_vac = surface - surface_nonvac

    # sanity check plot
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_wireframe(surface[:,:,0], surface[:,:,1], surface[:,:,2], alpha=0.2) # ours
    ax.plot_wireframe(surface_nonvac[:,:,0], surface_nonvac[:,:,1], surface_nonvac[:,:,2], alpha=0.2) # ours
    plt.show()

def test_surface_autodiff():
    """
    Test autodifferentation of surface
    """
    ntheta = 256
    r = 0.1
    
    stel = Qsc.from_paper("precise QA", nphi=31, I2=10.0, p2=-1e6, order='r2')

    surface = stel.surface(r=r, ntheta=ntheta) # (3, nphi)
    loss = torch.mean(surface**2)

    # compute the gradient using autodiff
    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]

    # check rc gradient with finite difference
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        surface = stel.surface(r=r, ntheta=ntheta) # (3, nphi)
        return torch.mean(surface**2)
    dloss_by_drc_fd = finite_difference_torch(fd_obj, torch.clone(stel.rc.detach()), 1e-8)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    assert err.item() < 1e-5, f"dloss/drc finite difference check failed: {err.item()}"

    # check zs gradient with finite difference
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        surface = stel.surface(r=r, ntheta=ntheta) # (3, nphi)
        return torch.mean(surface**2)
    dloss_by_dzs_fd = finite_difference_torch(fd_obj, torch.clone(stel.zs.detach()), 1e-7)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    assert err.item() < 1e-5, f"dloss/dzs finite difference check failed: {err.item()}"

    # check etabar gradient with finite difference
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        surface = stel.surface(r=r, ntheta=ntheta) # (3, nphi)
        return torch.mean(surface**2)
    dloss_by_detabar_fd = finite_difference_torch(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-6)
    err = torch.abs(dloss_by_detabar - dloss_by_detabar_fd)
    assert err.item() < 1e-5, f"dloss/detabar finite difference check failed: {err.item()}"


def test_normal_autodiff():
    """
    Test autodifferentation of normal
    """
    ntheta = 256
    r = 0.1
    
    stel = Qsc.from_paper("precise QA", nphi=31, I2=10.0, p2=-1e4, order='r2')

    normal = stel.surface_normal(r=r, ntheta=ntheta) # (3, nphi)
    loss = torch.mean(normal**2)/100

    # compute the gradient using autodiff
    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]

    # check rc gradient with finite difference
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        normal = stel.surface_normal(r=r, ntheta=ntheta) # (3, nphi)
        return torch.mean(normal**2)/100
    dloss_by_drc_fd = finite_difference_torch(fd_obj, torch.clone(stel.rc.detach()), 1e-8)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    assert err.item() < 1e-4, f"dloss/drc finite difference check failed: {err.item()}"

    # check zs gradient with finite difference
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        normal = stel.surface_normal(r=r, ntheta=ntheta) # (3, nphi)
        return torch.mean(normal**2)/100
    dloss_by_dzs_fd = finite_difference_torch(fd_obj, torch.clone(stel.zs.detach()), 1e-8)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    assert err.item() < 1e-4, f"dloss/dzs finite difference check failed: {err.item()}"

    # check etabar gradient with finite difference
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        normal = stel.surface_normal(r=r, ntheta=ntheta) # (3, nphi)
        return torch.mean(normal**2)/100
    dloss_by_detabar_fd = finite_difference_torch(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-6)
    err = torch.abs(dloss_by_detabar - dloss_by_detabar_fd)
    assert err.item() < 1e-5, f"dloss/detabar finite difference check failed: {err.item()}"

def test_surface_area():
    """ Test surface area computation """

    # test the area of a torus with major radius R and minor radius r
    R = 1.0
    r = 0.1
    stel = Qsc(rc=[R,r], zs=[0,r], nphi=31, nfp=1, order='r3')
    a = 0.5 * (R-r)
    c = 0.5 * (R+r)
    ntheta = 32
    theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)[None, :]
    theta = torch.tensor(theta)
    dA = a * (c + a * torch.cos(theta))
    area_actual =  4 * np.pi**2 * a * c
    with patch.object(Qsc, 'surface_area_element', return_value=dA) as mock_method:
        area = stel.surface_area(r=r, ntheta=ntheta).item()
        assert abs(area - area_actual) < 1e-15, f"Surface area incorrect: {area}"
        mock_method.assert_called_once()

def test_surface_curvature():
    """ Test surface curvature computation """

    # test the curvature of elliptic flux surfaces
    stel = Qsc.from_paper("2022 QH nfp3 beta", order='r1')
    r = 0.1
    ntheta = 32
    curvature = stel.surface_theta_curvature(r=r, ntheta=ntheta).detach().numpy() # (nphi, ntheta)
    theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)[:, None]
    theta = torch.tensor(theta)

    # calculate the curvature of the ellipse analytically
    numerator = torch.abs(stel.X1c_untwisted * stel.Y1s_untwisted 
                          - stel.X1s_untwisted * stel.Y1c_untwisted
                          ) 
    denominator = r * torch.sqrt((stel.X1s_untwisted * torch.cos(theta) - stel.X1c_untwisted * torch.sin(theta))**2 + (stel.Y1s_untwisted * torch.cos(theta) - stel.Y1c_untwisted * torch.sin(theta))**2)**3
    curvature_actual =  numerator / denominator
    # relative err
    rel_err = np.abs(curvature - curvature_actual.T.detach().numpy()) / np.abs(curvature_actual.T.detach().numpy())
    err = np.max(rel_err)
    assert err < 1e-14, f"Surface curvature incorrect: {err}"


if __name__ == "__main__":
    test_surface()
    test_surface_tangents()
    test_second_derivative()
    test_surface_nonvac()
    test_surface_autodiff()
    test_normal_autodiff()
    test_surface_area()
    test_surface_curvature()