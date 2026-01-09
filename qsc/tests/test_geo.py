
import numpy as np
from qsc.qsc import Qsc
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from qsc.util import finite_difference_torch
from unittest.mock import patch
import time

def test_load_components():
    """Test the _load_components method of Qsc."""

    names = ["2022 QH nfp3 beta", "precise QA", "precise QH"]
    for name in names:
        stel = Qsc.from_paper(name, I2 = 1.0, p2=-1e5, order='r3')

        variables = ['X1c','Y1c','X1s','Y1s']
        if stel.order == 'r2':
            variables += ['X20','Y20','Z20','X2c','Y2c','Z2c','X2s','Y2s','Z2s']
        if stel.order == 'r3':
                variables += ['X3c1','X3s1','X3c3','X3s3','Y3c1','Y3s1','Y3c3','Y3s3','Z3c1','Z3s1','Z3c3','Z3s3']

        # test with vacuum_component = True
        components = stel._load_components(vacuum_component=True)
        for v in variables:
            assert torch.allclose(getattr(components, v+'_untwisted'), getattr(stel, v+'_vac_untwisted'), atol=1e-15), f"{v} vacuum component mismatch"

        # test with vacuum_component = False
        components = stel._load_components(vacuum_component=False)
        for v in variables:
            assert torch.allclose(getattr(components, v+'_untwisted'), getattr(stel, v+'_untwisted'), atol=1e-15), f"{v} component mismatch"

def test_surface():
    """
    Test the accuracy of the surface computation
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", order='r3')

    minor_radius = 0.1
    ntheta = 16

    # sanity check plot
    X,Y,Z,R = stel.get_boundary(r=minor_radius, ntheta=ntheta, nphi = stel.nphi, ntheta_fourier=256, mpol=25, ntor=25) # (nphi, ntheta)
    xyz = stel.surface(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_wireframe(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], alpha=0.2) # ours
    ax.plot_wireframe(X, Y, Z, alpha=0.2, color='orange') # qsc
    plt.show()

    # test the vacuum_component=True argument
    names = ["precise QA", "precise QH"]
    for name in names:
        stel = Qsc.from_paper(name, I2 = 1.0, p2=-1e5, order='r3')
        stel_vac = Qsc.from_paper(name, I2 = 0.0, p2=0.0, order='r3')
        xyz_vac_stel = stel.surface(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
        xyz_stel_vac = stel_vac.surface(r=minor_radius, ntheta=ntheta) # (nphi, ntheta, 3)
        xyz_vac_stel_vac = stel_vac.surface(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)

        # in vacuum, total vacuum solution and vacuum components should match
        assert torch.allclose(xyz_stel_vac, xyz_vac_stel_vac, atol=1e-14), "Vacuum surface mismatch in vacuum case"
        # vacuum component of nonvac field should match the total vacuum solution
        assert torch.allclose(xyz_vac_stel, xyz_stel_vac, atol=1e-14), "Vacuum surface mismatch between nonvac and vac case"

        # test the cache works
        v1 = stel.surface(r=minor_radius, ntheta=ntheta)
        t0 = time.time()
        v2 = stel.surface(r=minor_radius, ntheta=ntheta)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of surface failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing
        v1 = stel.surface(r=minor_radius, ntheta=ntheta, vacuum_component=True)
        t0 = time.time()
        v2 = stel.surface(r=minor_radius, ntheta=ntheta, vacuum_component=True)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of surface failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing

def test_surface_tangents():
    # set up the expansion
    stel = Qsc.from_paper("precise QA", nphi=513, order='r3')

    minor_radius = 0.1
    ntheta = 512
    xyz = stel.surface(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    dxyz_by_dvarphi = stel.dsurface_by_dvarphi(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    dxyz_by_dtheta = stel.dsurface_by_dtheta(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)
    dxyz_by_dr = stel.dsurface_by_dr(r=minor_radius, ntheta=ntheta).detach().numpy() # (nphi, ntheta, 3)

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

    # test the vacuum_component=True argument
    names = ["precise QH", "precise QA"]
    for name in names:
        stel = Qsc.from_paper(name, I2 = 1.0, p2=-1e5, order='r3')
        stel_vac = Qsc.from_paper(name, I2 = 0.0, p2=0.0, order='r3')

        dxyz_by_dvarphi_vac_stel = stel.dsurface_by_dvarphi(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
        dxyz_by_dvarphi_stel_vac = stel_vac.dsurface_by_dvarphi(r=minor_radius, ntheta=ntheta) # (nphi, ntheta, 3)
        dxyz_by_dvarphi_vac_stel_vac = stel_vac.dsurface_by_dvarphi(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
        # in vacuum, total vacuum solution and vacuum components should match
        assert torch.allclose(dxyz_by_dvarphi_stel_vac, dxyz_by_dvarphi_vac_stel_vac, atol=1e-14), "Vacuum dsurface_by_dvarphi mismatch in vacuum case"
        # vacuum component of nonvac field should match the total vacuum solution
        assert torch.allclose(dxyz_by_dvarphi_vac_stel, dxyz_by_dvarphi_stel_vac, atol=1e-14), "Vacuum dsurface_by_dvarphi mismatch between nonvac and vac case"

        dxyz_by_dtheta_vac_stel = stel.dsurface_by_dtheta(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
        dxyz_by_dtheta_stel_vac = stel_vac.dsurface_by_dtheta(r=minor_radius, ntheta=ntheta) # (nphi, ntheta, 3)
        dxyz_by_dtheta_vac_stel_vac = stel_vac.dsurface_by_dtheta(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
        # in vacuum, total vacuum solution and vacuum components should match
        assert torch.allclose(dxyz_by_dtheta_stel_vac, dxyz_by_dtheta_vac_stel_vac, atol=1e-14), "Vacuum dsurface_by_dtheta mismatch in vacuum case"
        # vacuum component of nonvac field should match the total vacuum solution
        assert torch.allclose(dxyz_by_dtheta_vac_stel, dxyz_by_dtheta_stel_vac, atol=1e-14), "Vacuum dsurface_by_dtheta mismatch between nonvac and vac case"

        dxyz_by_dr_vac_stel = stel.dsurface_by_dr(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
        dxyz_by_dr_stel_vac = stel_vac.dsurface_by_dr(r=minor_radius, ntheta=ntheta) # (nphi, ntheta, 3)
        dxyz_by_dr_vac_stel_vac = stel_vac.dsurface_by_dr(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
        # in vacuum, total vacuum solution and vacuum components should match
        assert torch.allclose(dxyz_by_dr_stel_vac, dxyz_by_dr_vac_stel_vac, atol=1e-14), "Vacuum dsurface_by_dr mismatch in vacuum case"
        # vacuum component of nonvac field should match the total vacuum solution
        assert torch.allclose(dxyz_by_dr_vac_stel, dxyz_by_dr_stel_vac, atol=1e-14), "Vacuum dsurface_by_dr mismatch between nonvac and vac case"

        # test the cache works
        v1 = stel.dsurface_by_dvarphi(r=minor_radius, ntheta=ntheta)
        t0 = time.time()
        v2 = stel.dsurface_by_dvarphi(r=minor_radius, ntheta=ntheta)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of dsurface_by_dvarphi failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing
        v1 = stel.dsurface_by_dvarphi(r=minor_radius, ntheta=ntheta, vacuum_component=True)
        t0 = time.time()
        v2 = stel.dsurface_by_dvarphi(r=minor_radius, ntheta=ntheta, vacuum_component=True)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of dsurface_by_dvarphi failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing

        # test the cache works
        v1 = stel.dsurface_by_dtheta(r=minor_radius, ntheta=ntheta)
        t0 = time.time()
        v2 = stel.dsurface_by_dtheta(r=minor_radius, ntheta=ntheta)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of dsurface_by_dtheta failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing
        v1 = stel.dsurface_by_dtheta(r=minor_radius, ntheta=ntheta, vacuum_component=True)
        t0 = time.time()
        v2 = stel.dsurface_by_dtheta(r=minor_radius, ntheta=ntheta, vacuum_component=True)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of dsurface_by_dtheta failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing
        
        # test the cache works
        v1 = stel.dsurface_by_dr(r=minor_radius, ntheta=ntheta)
        t0 = time.time()
        v2 = stel.dsurface_by_dr(r=minor_radius, ntheta=ntheta)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of dsurface_by_dr failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing
        v1 = stel.dsurface_by_dr(r=minor_radius, ntheta=ntheta, vacuum_component=True)
        t0 = time.time()
        v2 = stel.dsurface_by_dr(r=minor_radius, ntheta=ntheta, vacuum_component=True)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of dsurface_by_dr failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing

def test_surface_normal():
    """Test the surface_normal method."""

    minor_radius = 0.1
    ntheta=32
    # test the vacuum_component=True argument
    names = ["precise QH", "precise QA"]
    for name in names:
        # test the vacuum_component=True argument
        names = ["precise QH", "precise QA"]
        for name in names:
            stel = Qsc.from_paper(name, I2 = 1.0, p2=-1e5, order='r3')
            stel_vac = Qsc.from_paper(name, I2 = 0.0, p2=0.0, order='r3')

            normal_stel = stel.surface_normal(r=minor_radius, ntheta=ntheta, vacuum_component=False) # (nphi, ntheta, 3)
            normal_vac_stel = stel.surface_normal(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
            normal_stel_vac = stel_vac.surface_normal(r=minor_radius, ntheta=ntheta) # (nphi, ntheta, 3)
            normal_vac_stel_vac = stel_vac.surface_normal(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
            # in vacuum, total vacuum solution and vacuum components should match
            assert torch.allclose(normal_stel_vac, normal_vac_stel_vac, atol=1e-14), "Vacuum normal mismatch in vacuum case"
            # vacuum component of nonvac field should match the total vacuum solution
            assert torch.allclose(normal_vac_stel, normal_stel_vac, atol=1e-14), "Vacuum normal mismatch between nonvac and vac case"

            # test normal is orthogonal to tangent
            dxyz_by_dtheta = stel.dsurface_by_dtheta(r=minor_radius, ntheta=ntheta) # (nphi, ntheta, 3)
            err = torch.max(torch.abs(torch.sum(normal_stel * dxyz_by_dtheta,axis=-1)))
            print(err.detach().numpy())
            assert err < 1e-14, "normal not orthogonal to theta tangent"

            # test normal is orthogonal to tangent
            dxyz_by_dvarphi = stel.dsurface_by_dvarphi(r=minor_radius, ntheta=ntheta) # (nphi, ntheta, 3)
            err = torch.max(torch.abs(torch.sum(normal_stel * dxyz_by_dvarphi,axis=-1)))
            print(err.detach().numpy())
            assert err < 1e-14, "normal not orthogonal to varphi tangent"

            # test vacuum normal is orthogonal to vacuum tangent
            dxyz_by_dtheta = stel.dsurface_by_dtheta(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
            err = torch.max(torch.abs(torch.sum(normal_vac_stel * dxyz_by_dtheta,axis=-1)))
            print(err.detach().numpy())
            assert err < 1e-14, "vacuum normal not orthogonal to vacuum theta tangent"

            # test vacuum normal is orthogonal to vacuum tangent
            dxyz_by_dvarphi = stel.dsurface_by_dvarphi(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
            err = torch.max(torch.abs(torch.sum(normal_vac_stel * dxyz_by_dvarphi,axis=-1)))
            print(err.detach().numpy())
            assert err < 1e-14, "vacuum normal not orthogonal to vacuum varphi tangent"

            # test the cache works
            v1 = stel.surface_normal(r=minor_radius, ntheta=ntheta)
            t0 = time.time()
            v2 = stel.surface_normal(r=minor_radius, ntheta=ntheta)
            t1 = time.time()
            assert t1 - t0 < 1e-4, "Caching of surface_normal failed"
            np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
            del v1, v2 # for timing
            v1 = stel.surface_normal(r=minor_radius, ntheta=ntheta, vacuum_component=True)
            t0 = time.time()
            v2 = stel.surface_normal(r=minor_radius, ntheta=ntheta, vacuum_component=True)
            t1 = time.time()
            assert t1 - t0 < 1e-4, "Caching of surface_normal failed"
            np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
            del v1, v2 # for timing


def test_second_derivative():
    """ Test the accuracy of the d2surface_by_dthetatheta()."""
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

    # test the vacuum_component=True argument
    names = ["precise QH", "precise QA"]
    for name in names:
        stel = Qsc.from_paper(name, I2 = 1.0, p2=-1e5, order='r3')
        stel_vac = Qsc.from_paper(name, I2 = 0.0, p2=0.0, order='r3')

        d2theta_vac_stel = stel.d2surface_by_dthetatheta(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
        d2theta_stel_vac = stel_vac.d2surface_by_dthetatheta(r=minor_radius, ntheta=ntheta) # (nphi, ntheta, 3)
        d2theta_vac_stel_vac = stel_vac.d2surface_by_dthetatheta(r=minor_radius, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
        # in vacuum, total vacuum solution and vacuum components should match
        assert torch.allclose(d2theta_stel_vac, d2theta_vac_stel_vac, atol=1e-14), "d2surface_by_dthetatheta mismatch in vacuum case"
        # vacuum component of nonvac field should match the total vacuum solution
        assert torch.allclose(d2theta_vac_stel, d2theta_stel_vac, atol=1e-14), "Vacuum d2surface_by_dthetatheta mismatch between nonvac and vac case"

        # test the cache works
        v1 = stel.d2surface_by_dthetatheta(r=minor_radius, ntheta=ntheta)
        t0 = time.time()
        v2 = stel.d2surface_by_dthetatheta(r=minor_radius, ntheta=ntheta)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of d2surface_by_dthetatheta failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing
        v1 = stel.d2surface_by_dthetatheta(r=minor_radius, ntheta=ntheta, vacuum_component=True)
        t0 = time.time()
        v2 = stel.d2surface_by_dthetatheta(r=minor_radius, ntheta=ntheta, vacuum_component=True)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of d2surface_by_dthetatheta failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing
        
def test_surface_autodiff():
    """
    Test autodifferentation of surface().
    """
    ntheta = 256
    r = 0.1
    
    for vacuum_component in [True, False]:
        stel = Qsc.from_paper("precise QA", nphi=61, I2=0.1, p2=-1e5, order='r3')

        surface = stel.surface(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (3, nphi)
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
            surface = stel.surface(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (3, nphi)
            return torch.mean(surface**2)
        dloss_by_drc_fd = finite_difference_torch(fd_obj, torch.clone(stel.rc.detach()), 1e-8)
        err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
        assert err.item() < 1e-5, f"dloss/drc finite difference check failed: {err.item()}"

        # check zs gradient with finite difference
        def fd_obj(x):
            stel.zs.data = x
            stel.calculate()
            surface = stel.surface(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (3, nphi)
            return torch.mean(surface**2)
        dloss_by_dzs_fd = finite_difference_torch(fd_obj, torch.clone(stel.zs.detach()), 1e-7)
        err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
        assert err.item() < 1e-5, f"dloss/dzs finite difference check failed: {err.item()}"

        # check etabar gradient with finite difference
        def fd_obj(x):
            stel.etabar.data = x
            stel.calculate()
            surface = stel.surface(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (3, nphi)
            return torch.mean(surface**2)
        dloss_by_detabar_fd = finite_difference_torch(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-6)
        err = torch.abs(dloss_by_detabar - dloss_by_detabar_fd)
        assert err.item() < 1e-5, f"dloss/detabar finite difference check failed: {err.item()}"


def test_normal_autodiff():
    """
    Test autodifferentation of normal().
    """
    ntheta = 256
    r = 0.1
    
    stel = Qsc.from_paper("precise QA", nphi=61, I2=0.1, p2=-1e4, order='r3')

    for vacuum_component in [True, False]:
        normal = stel.surface_normal(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (3, nphi)
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
            normal = stel.surface_normal(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (3, nphi)
            return torch.mean(normal**2)/100
        dloss_by_drc_fd = finite_difference_torch(fd_obj, torch.clone(stel.rc.detach()), 1e-8)
        err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
        assert err.item() < 1e-4, f"dloss/drc finite difference check failed: {err.item()}"

        # check zs gradient with finite difference
        def fd_obj(x):
            stel.zs.data = x
            stel.calculate()
            normal = stel.surface_normal(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (3, nphi)
            return torch.mean(normal**2)/100
        dloss_by_dzs_fd = finite_difference_torch(fd_obj, torch.clone(stel.zs.detach()), 1e-8)
        err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
        assert err.item() < 1e-4, f"dloss/dzs finite difference check failed: {err.item()}"

        # check etabar gradient with finite difference
        def fd_obj(x):
            stel.etabar.data = x
            stel.calculate()
            normal = stel.surface_normal(r=r, ntheta=ntheta, vacuum_component=vacuum_component) # (3, nphi)
            return torch.mean(normal**2)/100
        dloss_by_detabar_fd = finite_difference_torch(fd_obj, torch.clone(torch.tensor([stel.etabar.detach()])), 1e-6)
        err = torch.abs(dloss_by_detabar - dloss_by_detabar_fd)
        assert err.item() < 1e-5, f"dloss/detabar finite difference check failed: {err.item()}"

def test_surface_area():
    """ Test surface_area() computation """

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

    # test the vacuum_component=True argument
    names = ["precise QH", "precise QA"]
    for name in names:
        stel = Qsc.from_paper(name, I2 = 1.0, p2=-1e5, order='r3')
        stel_vac = Qsc.from_paper(name, I2 = 0.0, p2=0.0, order='r3')

        area_vac_stel = stel.surface_area(r=r, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
        area_stel_vac = stel_vac.surface_area(r=r, ntheta=ntheta) # (nphi, ntheta, 3)
        area_vac_stel_vac = stel_vac.surface_area(r=r, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
        # in vacuum, total vacuum solution and vacuum components should match
        assert torch.allclose(area_stel_vac, area_vac_stel_vac, atol=1e-14), "surface_area mismatch in vacuum case"
        # vacuum component of nonvac field should match the total vacuum solution
        assert torch.allclose(area_vac_stel, area_stel_vac, atol=1e-14), "Vacuum surface_area mismatch between nonvac and vac case"
        
        # test the cache works
        v1 = stel.surface_area(r=r, ntheta=ntheta)
        t0 = time.time()
        v2 = stel.surface_area(r=r, ntheta=ntheta)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of surface_area failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing
        v1 = stel.surface_area(r=r, ntheta=ntheta, vacuum_component=True)
        t0 = time.time()
        v2 = stel.surface_area(r=r, ntheta=ntheta, vacuum_component=True)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of surface_area failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing

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

    # test the vacuum_component=True argument
    names = ["precise QH", "precise QA"]
    for name in names:
        stel = Qsc.from_paper(name, I2 = 1.0, p2=-1e5, order='r3')
        stel_vac = Qsc.from_paper(name, I2 = 0.0, p2=0.0, order='r3')

        curv_vac_stel = stel.surface_theta_curvature(r=r, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
        curv_stel_vac = stel_vac.surface_theta_curvature(r=r, ntheta=ntheta) # (nphi, ntheta, 3)
        curv_vac_stel_vac = stel_vac.surface_theta_curvature(r=r, ntheta=ntheta, vacuum_component=True) # (nphi, ntheta, 3)
        # in vacuum, total vacuum solution and vacuum components should match
        assert torch.allclose(curv_stel_vac, curv_vac_stel_vac, atol=1e-14), "surface_theta_curvature mismatch in vacuum case"
        # vacuum component of nonvac field should match the total vacuum solution
        assert torch.allclose(curv_vac_stel, curv_stel_vac, atol=1e-14), "Vacuum surface_theta_curvature mismatch between nonvac and vac case"

        # test the cache works
        v1 = stel.surface_theta_curvature(r=r, ntheta=ntheta)
        t0 = time.time()
        v2 = stel.surface_theta_curvature(r=r, ntheta=ntheta)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of surface_theta_curvature failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing
        v1 = stel.surface_theta_curvature(r=r, ntheta=ntheta, vacuum_component=True)
        t0 = time.time()
        v2 = stel.surface_theta_curvature(r=r, ntheta=ntheta, vacuum_component=True)
        t1 = time.time()
        assert t1 - t0 < 1e-4, "Caching of surface_theta_curvature failed"
        np.testing.assert_allclose(v1.detach().numpy(), v2.detach().numpy(), atol=1e-14)
        del v1, v2 # for timing



if __name__ == "__main__":
    test_load_components()
    test_surface()
    test_surface_tangents()
    test_surface_normal()
    test_second_derivative()
    test_surface_autodiff()
    test_normal_autodiff()
    test_surface_area()
    test_surface_curvature()