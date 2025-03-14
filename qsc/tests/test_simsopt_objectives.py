import numpy as np
from qsc.qsc import Qsc
import torch
import numpy as np
from simsopt.geo import create_equally_spaced_curves
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from qsc.simsopt_objectives import (FieldError, QscOptimizable, ExternalFieldError, GradExternalFieldError,
                                    IotaPenalty, AxisLengthPenalty, LGradB, B20Penalty, MagneticWellPenalty)
from scipy.optimize import approx_fprime
from qsc.util import finite_difference

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
    x[0] +=0.2
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
    stel = QscOptimizable.from_paper("precise QA", order='r1', nphi=511)
    stel.unfix_all()

    # from simsopt.geo import plot as sms_plot
    # import matplotlib.pyplot as plt
    # ax = plt.figure().add_subplot(projection='3d')
    # xyz = stel.XYZ0.detach().numpy() 
    # ax.plot(xyz[0],xyz[1],xyz[2])
    # sms_plot(coils, engine="matplotlib", ax=ax, close=True, show=False)
    # plt.show()

    fe = ExternalFieldError(biot_savart, stel, r=0.3, ntheta=256, ntarget=32)

    # modify some of the axis parameters
    fe.fix_all()
    biot_savart.fix_all()
    stel.unfix_all()
    stel.unfix('rc(0)')
    stel.unfix('rc(1)')
    x = fe.x
    x[0] +=0.2
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
    dfe_by_dqsc_fd = finite_difference(fun, x, 1e-6)
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
    stel = QscOptimizable.from_paper("precise QA", order='r1', nphi=257)

    fe = GradExternalFieldError(biot_savart, stel, r=0.3, ntheta=256, ntarget=16)

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
        dfe_by_dqsc_fd = finite_difference(fun, x, 1e-8)
    err = np.max(np.abs(dfe_by_dqsc_fd - dfe_by_dqsc))
    print(err)
    assert err < 1e-5, "FAIL: qsc derivatives are incorrect"

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

def test_LGradB():
    # set up the expansion
    stel = QscOptimizable.from_paper("precise QA", order='r1', nphi=99)
    ip = LGradB(stel)
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
    assert err < 1e-5, "FAIL: qsc derivatives are incorrect"

def test_B20Penalty():
    # set up the expansion
    stel = QscOptimizable.from_paper("precise QA", order='r2', nphi=99)
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


if __name__ == "__main__":
    test_FieldError()
    test_ExternalFieldError()
    test_GradExternalFieldError()
    test_IotaPenalty()
    test_AxisLengthPenalty()
    test_LGradB()
    test_B20Penalty()
    test_MagneticWellPenalty()