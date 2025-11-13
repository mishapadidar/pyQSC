import numpy as np
from qsc.qsc import Qsc
import torch
import numpy as np
from simsopt.geo import create_equally_spaced_curves
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from qsc.simsopt_objectives import (FieldError, QscOptimizable, ExternalFieldError, GradExternalFieldError,
                                    IotaPenalty, AxisLengthPenalty, GradBPenalty, GradGradBPenalty, B20Penalty, MagneticWellPenalty,
                                    GradFieldError, AxisArcLengthVariation, SurfaceSelfIntersectionPenalty,
                                    PressurePenalty, CurveAxisDistancePenalty, ThetaCurvaturePenalty, Iota, AxisLength, SquaredFlux)
from qsc.util import finite_difference

def test_save_load():
    """Test saving and loading a QscOptimizable object with Simsopt."""
    from simsopt._core import save, load

    # stel = QscOptimizable.from_paper("precise QA", order='r1', nphi=71, sG=-1, spsi=-1, B0 = 2)
    stel = QscOptimizable(rc = [1.0, 0.0, 0.1], zs=[0.0, 0.0], zc=[0.0, 0.0, 1e-4], p2=1e4, I2=1e-7, 
                          B2c=1e-6, B2s=1e-4, order='r1', nphi=71, sG=-1, spsi=-1, B0 = 2)

    stel.unfix_all()
    stel.save("savetest.json")
    stel2 = load("savetest.json")

    assert np.allclose(stel.x - stel2.x, 0.0, atol=1e-15), "loaded dofs do not match saved"
    assert stel.nfp == stel2.nfp, "loaded nfp do not match saved value"
    assert stel.B0 == stel2.B0, "loaded B0 do not match saved value"
    assert stel.sigma0 == stel2.sigma0, "loaded sigma0 do not match saved value"
    assert stel.sG == stel2.sG, "loaded sG do not match saved value"
    assert stel.spsi == stel2.spsi, "loaded spsi do not match saved value"
    assert stel.nphi == stel2.nphi, "loaded nphi do not match saved value"
    assert stel.order == stel2.order, "loaded order do not match saved value"

def test_get_scale():
    """ Test the get_scale() method of QscOptimizable."""

    # default case
    stel = QscOptimizable(rc=[1, 0], zs=[0, 0], order='r1', nphi=17)
    scale = stel.get_scale()
    scale_actual = np.array([1, np.exp(-1), 1, np.exp(-1),1, np.exp(-1), 1, np.exp(-1), 1, 1, 1, 1, 1])
    print(scale - scale_actual)
    np.testing.assert_allclose(scale, scale_actual, rtol=1e-15, atol=1e-15)

    # case with specified scales and fixed dofs
    stel = QscOptimizable(rc=[1, 0], zs=[0, 0], order='r1', nphi=17)
    stel.fix('rc(0)')
    stel.fix('etabar')
    scale = stel.get_scale(**{'p2':5.0, 'zs(1)':7.0})
    scale_actual = np.array([np.exp(-1), 1, 7.0, 1, np.exp(-1), 1, np.exp(-1), 1, 1, 5.0, 1])
    print(scale - scale_actual)
    np.testing.assert_allclose(scale, scale_actual, rtol=1e-15, atol=1e-15)

def test_FieldError():
    """
    The the FieldError class.
    """

    # configuration parameters
    ncoils = 4
    nfp = 2
    is_stellsym = True
    coil_major_radius = 1.0
    coil_minor_radius = 0.5
    coil_n_fourier_modes = 3
    coil_current = 100000.0 

    # initialize coils
    base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=is_stellsym, R0=coil_major_radius,
                                            R1=coil_minor_radius, order=coil_n_fourier_modes)
    base_currents = [Current(1.0) * coil_current for i in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=is_stellsym)
    biot_savart = BiotSavart(coils)

    # set up the expansion
    stel = QscOptimizable.from_paper("precise QA", order='r1')

    # from simsopt.geo import plot as sms_plot
    # import matplotlib.pyplot as plt
    # ax = plt.figure().add_subplot(projection='3d')
    # xyz = stel.XYZ0.detach().numpy() 
    # ax.plot(xyz[0],xyz[1],xyz[2])
    # sms_plot(coils, engine="matplotlib", ax=ax, close=True, show=False)
    # plt.show()

    fe = FieldError(biot_savart, stel)

    # modify some of the axis parameters
    fe.fix_all()
    biot_savart.fix_all()
    stel.unfix_all()
    stel.unfix('rc(0)')
    stel.unfix('rc(1)')
    x = fe.x
    x[0] +=0.03
    x[1] +=0.05
    fe.x = x

    # keep the base point for finite-differences
    fe.unfix_all()
    x0 = fe.x

    # compute derivatives
    fe.unfix_all()
    partials = fe.dfield_error()
    dfe_by_dbs = partials(biot_savart)
    dfe_by_dqsc = partials(stel)

    # check derivative w.r.t. coil dofs w/ finite difference
    fe.unfix_all()
    fe.x = x0
    fe.fix_all()
    biot_savart.unfix_all()
    x = biot_savart.x
    def fun(x):
        biot_savart.x = x
        return fe.field_error().detach().numpy()
    # dfe_by_dbs_fd = approx_fprime(x, fun, epsilon=1e-1)
    dfe_by_dbs_fd = finite_difference(fun, x, 1e-4)
    err = np.max(np.abs(dfe_by_dbs_fd - dfe_by_dbs))
    print(err)
    assert err < 1e-5, "FAIL: coil derivatives are incorrect"

    # check derivative w.r.t. axis dofs w/ finite difference
    fe.unfix_all()
    fe.x = x0
    fe.fix_all()
    stel.unfix_all()
    x = stel.x
    def fun(x):
        stel.x = x
        return fe.field_error().detach().numpy()
    dfe_by_dqsc_fd = finite_difference(fun, x, 1e-4)
    err = np.max(np.abs(dfe_by_dqsc_fd - dfe_by_dqsc))
    print(err)
    assert err < 1e-5, "FAIL: qsc derivatives are incorrect"

def test_GradFieldError():
    """
    The the FieldError class.
    """

    # configuration parameters
    ncoils = 4
    nfp = 2
    is_stellsym = True
    coil_major_radius = 1.0
    coil_minor_radius = 0.5
    coil_n_fourier_modes = 3
    coil_current = 100000.0 

    # initialize coils
    base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=is_stellsym, R0=coil_major_radius,
                                            R1=coil_minor_radius, order=coil_n_fourier_modes)
    base_currents = [Current(1.0) * coil_current for i in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=is_stellsym)
    biot_savart = BiotSavart(coils)

    # set up the expansion
    stel = QscOptimizable.from_paper("precise QA", order='r1')

    # from simsopt.geo import plot as sms_plot
    # import matplotlib.pyplot as plt
    # ax = plt.figure().add_subplot(projection='3d')
    # xyz = stel.XYZ0.detach().numpy() 
    # ax.plot(xyz[0],xyz[1],xyz[2])
    # sms_plot(coils, engine="matplotlib", ax=ax, close=True, show=False)
    # plt.show()

    fe = GradFieldError(biot_savart, stel)

    # modify some of the axis parameters
    fe.fix_all()
    biot_savart.fix_all()
    stel.unfix_all()
    stel.unfix('rc(0)')
    stel.unfix('rc(1)')
    x = fe.x
    x[0] +=0.03
    x[1] +=0.05
    fe.x = x

    # keep the base point for finite-differences
    fe.unfix_all()
    x0 = fe.x

    # compute derivatives
    fe.unfix_all()
    partials = fe.dfield_error()
    dfe_by_dbs = partials(biot_savart)
    dfe_by_dqsc = partials(stel)

    # check derivative w.r.t. coil dofs w/ finite difference
    fe.unfix_all()
    fe.x = x0
    fe.fix_all()
    biot_savart.unfix_all()
    x = biot_savart.x
    def fun(x):
        biot_savart.x = x
        return fe.field_error().detach().numpy()
    # dfe_by_dbs_fd = approx_fprime(x, fun, epsilon=1e-1)
    dfe_by_dbs_fd = finite_difference(fun, x, 1e-7)
    err = np.max(np.abs(dfe_by_dbs_fd - dfe_by_dbs))
    print(err)
    assert err < 1e-5, "FAIL: coil derivatives are incorrect"

    # check derivative w.r.t. axis dofs w/ finite difference
    fe.unfix_all()
    fe.x = x0
    fe.fix_all()
    stel.unfix_all()
    x = stel.x
    def fun(x):
        stel.x = x
        return fe.field_error().detach().numpy()
    dfe_by_dqsc_fd = finite_difference(fun, x, 1e-8)
    err = np.max(np.abs(dfe_by_dqsc_fd - dfe_by_dqsc))
    print(err)
    assert err < 1e-4, "FAIL: qsc derivatives are incorrect"


def test_ExternalFieldError():
    """
    Test the ExternalFieldError class.
    """

    # configuration parameters
    ncoils = 4
    nfp = 2
    is_stellsym = True
    coil_major_radius = 1.0
    coil_minor_radius = 0.5
    coil_n_fourier_modes = 3
    coil_current = 100000.0 

    # initialize coils
    base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=is_stellsym, R0=coil_major_radius,
                                            R1=coil_minor_radius, order=coil_n_fourier_modes)
    base_currents = [Current(1.0) * coil_current for i in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=is_stellsym)
    biot_savart = BiotSavart(coils)

    # set up the expansion
    # Make sure self-intersection doesnt exist: otherwise we lose accuracy!
    r = 0.1
    p2 = -1e5
    stel = QscOptimizable.from_paper("precise QA", order='r2', p2=-1e5, nphi=61)
    fe = ExternalFieldError(biot_savart, stel, r=0.1, ntheta=256, nphi=256)
    stel.unfix_all()

    # # Sanity check plot: no self-intersections
    # import matplotlib.pyplot as plt
    # ax = plt.figure().add_subplot(projection='3d')
    # xyz = stel.surface(r=0.1, ntheta=256).detach().numpy() # (nphi, ntheta, 3)
    # ax.plot_surface(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], color='lightgray', alpha=0.3)
    # ax.legend()
    # plt.show()

    # keep the base point for finite-differences
    fe.unfix_all()
    x0 = fe.x

    # compute derivatives
    fe.unfix_all()
    partials = fe.dfield_error()
    dfe_by_dbs = partials(biot_savart)
    dfe_by_dqsc = partials(stel)

    # check derivative w.r.t. coil dofs w/ finite difference
    fe.unfix_all()
    fe.x = x0
    fe.fix_all()
    biot_savart.unfix_all()
    x = biot_savart.x
    def fun(x):
        biot_savart.x = x
        return fe.field_error().detach().numpy()
    # dfe_by_dbs_fd = approx_fprime(x, fun, epsilon=1e-1)
    dfe_by_dbs_fd = finite_difference(fun, x, 1e-5)
    err = np.max(np.abs(dfe_by_dbs_fd - dfe_by_dbs))
    print(err)
    assert err < 1e-5, "FAIL: coil derivatives are incorrect"

    # check derivative w.r.t. axis dofs w/ finite difference
    fe.unfix_all()
    fe.x = x0
    fe.fix_all()
    stel.unfix_all()
    x = stel.x
    def fun(x):
        stel.x = x
        return fe.field_error().detach().numpy()
    dfe_by_dqsc_fd = finite_difference(fun, x, 1e-9)
    err = np.max(np.abs(dfe_by_dqsc_fd - dfe_by_dqsc))
    print(err)
    assert err < 1e-5, "FAIL: qsc derivatives are incorrect"


def test_GradExternalFieldError():
    """
    Test the GradExternalFieldError class.
    """

    # configuration parameters
    ncoils = 2
    nfp = 2
    is_stellsym = True
    coil_major_radius = 1.0
    coil_minor_radius = 0.5
    coil_n_fourier_modes = 2
    coil_current = 100000.0 

    # initialize coils
    base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=is_stellsym, R0=coil_major_radius,
                                            R1=coil_minor_radius, order=coil_n_fourier_modes)
    base_currents = [Current(1.0) * coil_current for i in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=is_stellsym)
    biot_savart = BiotSavart(coils)

    # set up the expansion
    stel = QscOptimizable.from_paper("precise QA", order='r2', p2=1e-5, nphi=257)
    fe = GradExternalFieldError(biot_savart, stel, r=0.1, ntheta=256)

    # keep the base point for finite-differences
    fe.unfix_all()
    x0 = fe.x

    # compute derivatives
    fe.unfix_all()
    partials = fe.dfield_error()
    dfe_by_dbs = partials(biot_savart)
    dfe_by_dqsc = partials(stel)

    # check derivative w.r.t. coil dofs w/ finite difference
    fe.unfix_all()
    fe.x = x0
    fe.fix_all()
    biot_savart.unfix_all()
    x = biot_savart.x
    def fun(x):
        biot_savart.x = x
        return fe.field_error().detach().numpy()
    # dfe_by_dbs_fd = approx_fprime(x, fun, epsilon=1e-1)
    dfe_by_dbs_fd = finite_difference(fun, x, 1e-6)
    err = np.max(np.abs(dfe_by_dbs_fd - dfe_by_dbs))
    print(err)
    assert err < 1e-5, "FAIL: coil derivatives are incorrect"

    # check derivative w.r.t. axis dofs w/ finite difference
    fe.unfix_all()
    fe.x = x0
    fe.fix_all()
    stel.unfix_all()
    x = stel.x
    def fun(x):
        stel.x = x
        return fe.field_error().detach().numpy()
    with torch.no_grad():
        dfe_by_dqsc_fd = finite_difference(fun, x, 1e-9)
    err = np.max(np.abs(dfe_by_dqsc_fd - dfe_by_dqsc))
    print(err)
    assert err < 1e-4, "FAIL: qsc derivatives are incorrect"

def test_IotaPenalty():
        # set up the expansion
    stel = QscOptimizable.from_paper("precise QA", order='r1', nphi=511)
    ip = IotaPenalty(stel, 0.6)
    stel.unfix_all()
    ip.unfix_all()
    x0 = ip.x

    # compute derivatives
    dJ_by_dqsc = ip.dJ()

    # check derivative w.r.t. axis dofs w/ finite difference
    def fun(x):
        stel.x = x
        return ip.J()
    dfe_by_dqsc_fd = finite_difference(fun, x0, 1e-6)
    err = np.max(np.abs(dfe_by_dqsc_fd - dJ_by_dqsc))
    print(err)
    assert err < 1e-5, "FAIL: qsc derivatives are incorrect"

def test_Iota():
    # set up the expansion
    stel = QscOptimizable.from_paper("precise QA", order='r1', nphi=511)
    ip = Iota(stel)
    stel.unfix_all()
    ip.unfix_all()
    x0 = ip.x
    assert np.allclose(ip.J(),stel.iota.detach().numpy().item(), atol=1e-14), "Iota objective value does not match iota property"

    # compute derivatives
    dJ_by_dqsc = ip.dJ()

    # check derivative w.r.t. axis dofs w/ finite difference
    def fun(x):
        stel.x = x
        return ip.J()
    dfe_by_dqsc_fd = finite_difference(fun, x0, 1e-6)
    err = np.max(np.abs(dfe_by_dqsc_fd - dJ_by_dqsc))
    print(err)
    assert err < 1e-5, "FAIL: qsc derivatives are incorrect"


def test_AxisLengthPenalty():
    # set up the expansion
    stel = QscOptimizable.from_paper("precise QA", order='r1', nphi=99)
    ip = AxisLengthPenalty(stel, 1.221)
    stel.unfix_all()
    ip.unfix_all()
    x0 = ip.x

    # compute derivatives
    dJ_by_dqsc = ip.dJ()

    # check derivative w.r.t. axis dofs w/ finite difference
    def fun(x):
        stel.x = x
        return ip.J()
    dfe_by_dqsc_fd = finite_difference(fun, x0, 1e-4)
    err = np.max(np.abs(dfe_by_dqsc_fd - dJ_by_dqsc))
    print(err)
    assert err < 1e-5, "FAIL: qsc derivatives are incorrect"

def test_AxisLength():
    # set up the expansion
    stel = QscOptimizable.from_paper("precise QA", order='r1', nphi=511)
    ip = AxisLength(stel)
    stel.unfix_all()
    ip.unfix_all()
    x0 = ip.x
    assert np.allclose(ip.J(),stel.axis_length.detach().numpy().item(), atol=1e-14), "AxisLength objective value does not match axis_length property"

    # compute derivatives
    dJ_by_dqsc = ip.dJ()

    # check derivative w.r.t. axis dofs w/ finite difference
    def fun(x):
        stel.x = x
        return ip.J()
    dfe_by_dqsc_fd = finite_difference(fun, x0, 1e-6)
    err = np.max(np.abs(dfe_by_dqsc_fd - dJ_by_dqsc))
    print(err)
    assert err < 1e-5, "FAIL: qsc derivatives are incorrect"

def test_GradBPenalty():
    # set up the expansion
    stel = QscOptimizable.from_paper("2022 QH nfp3 beta", order='r2', nphi=99)
    ip = GradBPenalty(stel)
    stel.unfix_all()
    ip.unfix_all()
    x0 = ip.x

    # compute derivatives
    dJ_by_dqsc = ip.dJ()

    # check derivative w.r.t. axis dofs w/ finite difference
    def fun(x):
        stel.x = x
        return ip.J()
    dfe_by_dqsc_fd = finite_difference(fun, x0, 1e-7)
    err = np.max(np.abs(dfe_by_dqsc_fd - dJ_by_dqsc))
    print(err)
    assert err < 1e-5, "FAIL: GradBPenalty derivatives are incorrect"

def test_GradGradBPenalty():
    # set up the expansion
    stel = QscOptimizable.from_paper("2022 QH nfp3 beta", order='r2', nphi=131)
    ip = GradGradBPenalty(stel)
    stel.unfix_all()
    ip.unfix_all()
    x0 = ip.x

    # compute derivatives
    dJ_by_dqsc = ip.dJ()

    # check derivative w.r.t. axis dofs w/ finite difference
    def fun(x):
        stel.x = x
        return ip.J()

    """ do a taylor test"""
    

    # compute central difference derivative
    h_values = [1e-7 / (2**i) for i in range(5)]
    errors = []
    for h in h_values:
        dfe_by_dqsc_fd = finite_difference(fun, x0, h)
        err = np.max(np.abs(dfe_by_dqsc_fd - dJ_by_dqsc))
        errors.append(err)

    # error should be quadratic for central difference
    log_errors = np.log(errors)
    log_h = np.log(h_values)
    slope = np.polyfit(log_h, log_errors, 1)[0]
    assert np.abs(slope - 2) < 1e-3, "FAIL: GradGradBPenalty derivatives are incorrect"

def test_B20Penalty():
    # set up the expansion
    stel = QscOptimizable.from_paper("2022 QH nfp3 beta", order='r2', nphi=99)
    ip = B20Penalty(stel)
    stel.unfix_all()
    ip.unfix_all()
    x0 = ip.x

    # compute derivatives
    dJ_by_dqsc = ip.dJ()

    # check derivative w.r.t. axis dofs w/ finite difference
    def fun(x):
        stel.x = x
        return ip.J()
    dfe_by_dqsc_fd = finite_difference(fun, x0, 1e-8)
    err = np.max(np.abs(dfe_by_dqsc_fd - dJ_by_dqsc))
    print(err)
    assert err < 1e-5, "FAIL: qsc derivatives are incorrect"

def test_MagneticWellPenalty():
    # set up the expansion
    stel = QscOptimizable.from_paper("precise QA", order='r2', nphi=99)
    ip = MagneticWellPenalty(stel, well_target=-100)
    stel.unfix_all()
    ip.unfix_all()
    x0 = ip.x

    # compute derivatives
    dJ_by_dqsc = ip.dJ()

    # check derivative w.r.t. axis dofs w/ finite difference
    def fun(x):
        stel.x = x
        return ip.J()
    dfe_by_dqsc_fd = finite_difference(fun, x0, 1e-9)
    err = np.max(np.abs(dfe_by_dqsc_fd - dJ_by_dqsc))
    print(err)
    assert err < 1e-4, "FAIL: qsc derivatives are incorrect"

def test_AxisArcLengthVariationPenalty():
    # set up the expansion
    stel = QscOptimizable.from_paper("precise QA", order='r2', nphi=99)
    ip = AxisArcLengthVariation(stel)
    stel.unfix_all()
    ip.unfix_all()
    x0 = ip.x

    # compute derivatives
    dJ_by_dqsc = ip.dJ()

    # check derivative w.r.t. axis dofs w/ finite difference
    def fun(x):
        stel.x = x
        return ip.J()
    dfe_by_dqsc_fd = finite_difference(fun, x0, 1e-3)
    err = np.max(np.abs(dfe_by_dqsc_fd - dJ_by_dqsc))
    print(err)
    assert err < 1e-4, "FAIL: qsc derivatives are incorrect"

def test_SurfaceSelfIntersectionPenalty():
    # use larger |p2| so that the surface is self-intersecting
    stel = QscOptimizable.from_paper("precise QA", order='r2', p2=-1.8e6, nphi=99)
    r = 0.1
    ip = SurfaceSelfIntersectionPenalty(stel, r=r, ntheta=32)
    stel.unfix_all()
    ip.unfix_all()
    x0 = ip.x

    # compute derivatives
    dJ_by_dqsc = ip.dJ()

    # check derivative w.r.t. axis dofs w/ finite difference
    def fun(x):
        stel.x = x
        return ip.J()
    dfe_by_dqsc_fd = finite_difference(fun, x0, 1e-9)
    err = np.max(np.abs(dfe_by_dqsc_fd - dJ_by_dqsc))
    print(err)
    assert err < 1e-4, "FAIL: qsc derivatives are incorrect"

def test_PressurePenalty():
    # set up the expansion
    stel = QscOptimizable.from_paper("precise QA", order='r2', p2=-1e5, nphi=99)
    ip = PressurePenalty(stel, -1e6)
    stel.unfix_all()
    ip.unfix_all()
    x0 = ip.x

    # compute derivatives
    dJ_by_dqsc = ip.dJ()

    # check derivative w.r.t. axis dofs w/ finite difference
    def fun(x):
        stel.x = x
        return ip.J()
    dip_by_dqsc_fd = finite_difference(fun, x0, 1e-4)
    err = np.max(np.abs(dip_by_dqsc_fd - dJ_by_dqsc))
    print(err)
    assert err < 1e-5, "FAIL: qsc derivatives are incorrect"

def test_CurveAxisDistancePenalty():

    """
    The the CurveAxisDistancePenalty class.
    """

    # configuration parameters
    ncoils = 2
    nfp = 2
    is_stellsym = True
    coil_major_radius = 1.0
    coil_minor_radius = 0.2
    coil_n_fourier_modes = 3
    coil_current = 100000.0 

    # initialize coils
    base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=is_stellsym, R0=coil_major_radius,
                                            R1=coil_minor_radius, order=coil_n_fourier_modes)

    # set up the expansion
    stel = QscOptimizable.from_paper("precise QA", order='r2')

    curve = base_curves[0]
    dmin = 0.3
    fe = CurveAxisDistancePenalty(curve, stel, minimum_distance=dmin)

    # modify some of the axis parameters
    fe.fix_all()
    stel.unfix_all()
    stel.unfix('rc(0)')
    stel.unfix('rc(1)')
    x = fe.x
    x[0] +=0.01
    x[1] +=0.01
    fe.x = x

    # modify some curve parameters
    curve.unfix_all()
    x = curve.x
    x[1] += 0.01
    x[2] += 0.01
    x[3] += 0.5
    curve.x = x

    # from simsopt.geo import plot as sms_plot
    # import matplotlib.pyplot as plt
    # ax = plt.figure().add_subplot(projection='3d')
    # xyz = stel.XYZ0.detach().numpy() 
    # ax.plot(xyz[0],xyz[1],xyz[2])
    # sms_plot(base_curves, engine="matplotlib", ax=ax, close=True, show=False)
    # plt.show()

    # keep the base point for finite-differences
    fe.unfix_all()
    x0 = fe.x

    # compute derivatives
    fe.unfix_all()
    partials = fe.dobj()
    dfe_by_dbs = partials(curve)
    dfe_by_dqsc = partials(stel)

    # check derivative w.r.t. coil dofs w/ finite difference
    fe.unfix_all()
    fe.x = x0
    fe.fix_all()
    curve.unfix_all()
    x = curve.x
    def fun(x):
        curve.x = x
        return fe.obj().detach().numpy()
    # dfe_by_dbs_fd = approx_fprime(x, fun, epsilon=1e-1)
    dfe_by_dbs_fd = finite_difference(fun, x, 1e-3)
    err = np.max(np.abs(dfe_by_dbs_fd - dfe_by_dbs))
    assert err < 1e-5, "FAIL: coil derivatives of CurveAxisDistancePenalty are incorrect"

    # check derivative w.r.t. axis dofs w/ finite difference
    fe.unfix_all()
    fe.x = x0
    fe.fix_all()
    stel.unfix_all()
    x = stel.x
    def fun(x):
        stel.x = x
        return fe.obj().detach().numpy()
    dfe_by_dqsc_fd = finite_difference(fun, x, 1e-3)
    err = np.max(np.abs(dfe_by_dqsc_fd - dfe_by_dqsc))
    assert err < 1e-5, "FAIL: qsc derivatives of CurveAxisDistancePenalty are incorrect"

    # make sure shortest distance is calculated correctly
    from simsopt.geo import CurveXYZFourier
    curve = CurveXYZFourier(32, 1)
    curve.x = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.53, 0.0, 0.0])
    rc = np.array([1.0, 0.0, 0.0])
    zs = np.array([0.0, 0.0, 0.0])
    stel = QscOptimizable(rc=rc, zs=zs, nphi=32)
    fe = CurveAxisDistancePenalty(curve, stel, minimum_distance=0.1)
    assert np.abs(fe.shortest_distance().detach().numpy().item() - 0.53) < 1e-14, "FAIL: distance calculation is incorrect"

def test_ThetaCurvaturePenalty():
    """ Test the ThetaCurvaturePenalty class"""

    # # test the objective value for elliptic cross sections
    # minor_radius = 0.1
    # ntheta = 32
    # stel = QscOptimizable.from_paper("2022 QH nfp3 beta", order='r1', nphi=99)
    # ip = ThetaCurvaturePenalty(stel, minor_radius, ntheta = ntheta, kappa_target=kappa_target)

    # set up the expansion
    minor_radius = 0.1
    ntheta = 32
    kappa_target = 4/minor_radius
    stel = QscOptimizable.from_paper("2022 QH nfp3 beta")
    ip = ThetaCurvaturePenalty(stel, minor_radius, ntheta = ntheta, kappa_target=kappa_target)
    stel.unfix_all()

    # rescale the dofs to get better finite difference accuracy
    scale = stel.get_scale(**{'p2':stel.get('p2').item()})
    y0 = stel.x / scale
    dx_dy = scale

    # normalize the penalty so that it is O(1)
    ip = (1/ip.J()) * ip

    # compute derivatives
    dJ_by_dqsc = ip.dJ() * dx_dy

    # check derivative w.r.t. axis dofs w/ finite difference
    def fun(y):
        x = y * scale
        stel.x = x
        return ip.J()
    dip_by_dqsc_fd = finite_difference(fun, y0, 1e-8)
    err = np.max(np.abs(dip_by_dqsc_fd - dJ_by_dqsc))
    print(err)
    assert err < 1e-5, "FAIL: qsc derivatives are incorrect"


def test_SquaredFlux():
    """
    Test the SquaredFlux class.
    """

    # configuration parameters
    ncoils = 4
    nfp = 2
    is_stellsym = True
    coil_major_radius = 1.0
    coil_minor_radius = 0.5
    coil_n_fourier_modes = 3
    coil_current = 100000.0 

    # initialize coils
    base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=is_stellsym, R0=coil_major_radius,
                                            R1=coil_minor_radius, order=coil_n_fourier_modes, numquadpoints=128)
    base_currents = [Current(1.0) * coil_current for i in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=is_stellsym)
    biot_savart = BiotSavart(coils)

    # set up the expansion
    stel = QscOptimizable.from_paper("precise QH", p2=-1e3, I2=1e-3, nphi=61, order='r3')

    for vacuum_component in [True, False]:

        fe = SquaredFlux(biot_savart, stel, r=0.01, ntheta=32, vacuum_component=vacuum_component)
        stel.unfix_all()

        # keep the base point for finite-differences
        fe.unfix_all()
        x0 = fe.x

        # compute derivatives
        fe.unfix_all()
        partials = fe.dsquared_flux()
        dfe_by_dbs = partials(biot_savart)
        dfe_by_dqsc = partials(stel)

        # check derivative w.r.t. coil dofs w/ finite difference
        fe.unfix_all()
        fe.x = x0
        fe.fix_all()
        biot_savart.unfix_all()
        x = biot_savart.x
        def fun(x):
            biot_savart.x = x
            return fe.squared_flux().detach().numpy()
        dfe_by_dbs_fd = finite_difference(fun, x, 1e-4)
        err = np.max(np.abs(dfe_by_dbs_fd - dfe_by_dbs))
        print(err)
        assert err < 1e-5, "FAIL: coil derivatives are incorrect"

        # check derivative w.r.t. axis dofs w/ finite difference
        fe.unfix_all()
        fe.x = x0
        fe.fix_all()
        stel.unfix_all()
        x = stel.x
        def fun(x):
            stel.x = x
            return fe.squared_flux().detach().numpy()
        dfe_by_dqsc_fd = finite_difference(fun, x, 1e-8)
        err = np.max(np.abs(dfe_by_dqsc_fd - dfe_by_dqsc))
        print(err)
        assert err < 1e-5, "FAIL: qsc derivatives are incorrect"

    # test the vacuum_component=True argument
    names = ["precise QH", "precise QA"]
    for name in names:
        stel = QscOptimizable.from_paper(name, I2 = 1.0, p2=-1e5, order='r3')
        stel_vac = QscOptimizable.from_paper(name, I2 = 0.0, p2=0.0, order='r3')

        fe = SquaredFlux(biot_savart, stel, r=0.01, ntheta=32, vacuum_component=True)
        fe_vac = SquaredFlux(biot_savart, stel_vac, r=0.01, ntheta=32, vacuum_component=False)
        fe_vac_vac = SquaredFlux(biot_savart, stel_vac, r=0.01, ntheta=32, vacuum_component=True)

        # in vacuum, total vacuum solution and vacuum components should match
        assert np.allclose(fe_vac.J(), fe_vac_vac.J(), atol=1e-14), "J() mismatch in vacuum case"
        # vacuum component of nonvac field should match the total vacuum solution
        assert np.allclose(fe.J(), fe_vac.J(), atol=1e-14), "Vacuum J() mismatch between nonvac and vac case"

if __name__ == "__main__":
    test_save_load()
    test_get_scale()
    test_FieldError()
    test_GradFieldError()
    test_ExternalFieldError()
    test_GradExternalFieldError()
    test_IotaPenalty()
    test_Iota()
    test_AxisLengthPenalty()
    test_AxisLength()
    test_GradBPenalty()
    test_GradGradBPenalty()
    test_B20Penalty()
    test_MagneticWellPenalty()
    test_AxisArcLengthVariationPenalty()
    test_SurfaceSelfIntersectionPenalty()
    test_PressurePenalty()
    test_CurveAxisDistancePenalty()
    test_ThetaCurvaturePenalty()
    test_SquaredFlux()