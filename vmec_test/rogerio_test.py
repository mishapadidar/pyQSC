import os
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.mhd import Vmec
import glob
from qsc.qsc import Qsc
import matplotlib.pyplot as plt
from simsopt import load    

stel = Qsc.from_paper("precise QA", order='r2')
aspect_ratio_array = [4, 6, 8, 10, 20, 30, 40, 50]
# aspect_ratio_array = [30, 40, 50]

def optimize_coils(aspect_ratio=6):
    # Number of unique coil shapes, i.e. the number of coils per half field period:
    # (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
    ncoils = 6

    # Major radius for the initial circular coils:
    R0 = 1.0

    # Minor radius for the initial circular coils:
    R1 = 0.8

    # Number of Fourier modes describing each Cartesian component of each coil:
    order = 10

    # Weight on the curve lengths in the objective function. We use the `Weight`
    # class here to later easily adjust the scalar value and rerun the optimization
    # without having to rebuild the objective.
    LENGTH_WEIGHT = Weight(1e-14)

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    CC_THRESHOLD = 0.1
    CC_WEIGHT = 1000

    # Threshold and weight for the coil-to-surface distance penalty in the objective function:
    CS_THRESHOLD = 0.3
    CS_WEIGHT = 10

    # Threshold and weight for the curvature penalty in the objective function:
    CURVATURE_THRESHOLD = 10.
    CURVATURE_WEIGHT = 1e-8

    # Threshold and weight for the mean squared curvature penalty in the objective function:
    MSC_THRESHOLD = 10
    MSC_WEIGHT = 1e-8

    # Number of iterations to perform:
    MAXITER = 3000


    #######################################################
    # End of input parameters.
    #######################################################

    tag = "precise QA"

    print(f"#"*80)
    print(f"Aspect ratio: {aspect_ratio}")
    ff  = f"input.precise_QA_aspect_{aspect_ratio}"
    # stel.to_vmec(ff, r = stel.rc[0].detach().numpy()/aspect_ratio, ntheta=64)
    # stel.plot_boundary(r=stel.rc[0]/aspect_ratio)
    # exit()
    nphi = 64
    ntheta = 64
    vmec = Vmec(ff, range_surface="half period", nphi=nphi, ntheta=ntheta)
    s = vmec.boundary

    # # Initialize the boundary magnetic surface:
    # nphi = 32
    # ntheta = 32
    # s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

    # Create the initial coils:
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order,
                                            numquadpoints=128)
    base_currents = [Current(1)*1e5 for i in range(ncoils)]
    # Since the target field is zero, one possible solution is just to set all
    # currents to 0. To avoid the minimizer finding that solution, we fix one
    # of the currents:
    base_currents[0].fix_all()

    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(s.gamma().reshape((-1, 3)))

    curves = [c.curve for c in coils]
    # curves_to_vtk(curves, "./curves_init")
    # pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    # s.to_vtk("./surf_init", extra_data=pointData)

    # Define the individual terms objective function:
    Jf = SquaredFlux(s, bs, definition='normalized')
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]


    # Form the total objective function. To do this, we can exploit the
    # fact that Optimizable objects with J() and dJ() functions can be
    # multiplied by scalars and added:
    JF = (Jf \
        + LENGTH_WEIGHT * sum(Jls) \
        # + CC_WEIGHT * Jccdist \
        # + CS_WEIGHT * Jcsdist \
        # + CURVATURE_WEIGHT * sum(Jcs) \
        # + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
    )

    # We don't have a general interface in SIMSOPT for optimisation problems that
    # are not in least-squares form, so we write a little wrapper function that we
    # pass directly to scipy.optimize.minimize


    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        # jf = Jf.J()
        # Bbs = bs.B().reshape((nphi, ntheta, 3))
        # BdotN_over_B = np.mean(np.abs(np.sum(Bbs * s.unitnormal(), axis=2))/np.linalg.norm(Bbs, axis=2))
        # outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n/B⟩={BdotN_over_B:.1e}"
        # cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        # kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        # msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        # outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        # # outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        # outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        # print(outstr)
        return J, grad


    # print("""
    # ################################################################################
    # ### Perform a Taylor test ######################################################
    # ################################################################################
    # """)
    # f = fun
    # dofs = JF.x
    # np.random.seed(1)
    # h = np.random.uniform(size=dofs.shape)
    # J0, dJ0 = f(dofs)
    # dJh = sum(dJ0 * h)
    # for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    #     J1, _ = f(dofs + eps*h)
    #     J2, _ = f(dofs - eps*h)
    #     print("err", (J1-J2)/(2*eps) - dJh)

    print("""
    ################################################################################
    ### Run the optimisation #######################################################
    ################################################################################
    """)
    dofs = JF.x
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'disp': True}, tol=1e-15)
    # curves_to_vtk(curves, "./curves_opt_short")
    # pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    # s.to_vtk("./surf_opt_short", extra_data=pointData)


    # # We now use the result from the optimization as the initial guess for a
    # # subsequent optimization with reduced penalty for the coil length. This will
    # # result in slightly longer coils but smaller `B·n` on the surface.
    # dofs = res.x
    # LENGTH_WEIGHT *= 0.1
    # res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    # # curves_to_vtk(curves, "./curves_opt_long")
    # # pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    # # s.to_vtk("./surf_opt_long", extra_data=pointData)

    print('err')
    B_coil = bs.B().reshape((nphi, ntheta, 3))
    n = s.unitnormal()
    print(np.max(np.abs(np.sum(B_coil * n, axis=-1))))

    # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
    bs.unfix_all()
    # bs.save("./biot_savart_opt.json")
    # ff = f"./biot_savart_{name}_beta_{beta}_aspect_{aspect_ratio}_order_{expansion_order}.np"
    bs.save(f"./biot_savart_aspect_{aspect_ratio}.json")
    # ff = f"./biot_savart_aspect_{aspect_ratio}.np"
    # np.save(open(ff, "wb"), bs.x)

def test_coils():
    error_array_1 = []
    error_array_2 = []
    for i, aspect_ratio in enumerate(aspect_ratio_array):
        ff  = f"input.precise_QA_aspect_{aspect_ratio}"
        vmec = Vmec(ff, range_surface="half period", nphi=32, ntheta=32, verbose=False)
        vmec.boundary.to_vtk(f"surf_aspect_{aspect_ratio}")
        bs = load(f"./biot_savart_aspect_{aspect_ratio}.json")
        curves_to_vtk([coil.curve for coil in bs.coils], f"curves_aspect_{aspect_ratio}", close=True)
        R0 = stel.R0.detach().numpy()
        Z0 = stel.Z0.detach().numpy()
        phi = stel.phi.detach().numpy()
        points = np.array([R0*np.cos(phi), R0*np.sin(phi), Z0]).T
        bs.set_points(np.ascontiguousarray(points))
        axis_field = stel.Bfield_cartesian().detach().numpy().T/np.mean(np.linalg.norm(stel.Bfield_cartesian().detach().numpy().T, axis=-1))
        coil_field = bs.B()/np.mean(np.linalg.norm(bs.B(), axis=-1))
        B_error = axis_field - coil_field
        error_array_1.append(np.linalg.norm(B_error))
        error_array_2.append(np.max(np.abs(B_error)))
    # plt.plot(aspect_ratio_array, error_array_1)
    plt.plot(1/np.array(aspect_ratio_array), error_array_2)
    plt.plot(1/np.array(aspect_ratio_array), [0.8/ar**3 for ar in aspect_ratio_array], 'r--', label='1/aspect_ratio^3')
    plt.plot(1/np.array(aspect_ratio_array), [0.15/ar**2 for ar in aspect_ratio_array], 'k--', label='1/aspect_ratio^2')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    
    # ax = plt.figure().add_subplot(111, projection='3d')
    # ax.quiver(points[:, 0], points[:, 1], points[:, 2], axis_field[:, 0], axis_field[:, 1], axis_field[:, 2], color='r', normalize=True, length=0.1)
    # ax.quiver(points[:, 0], points[:, 1], points[:, 2], coil_field[:, 0], coil_field[:, 1], coil_field[:, 2], color='b', normalize=True, length=0.05)
    # plt.show()

def test_B_on_surface():

    error_array_1 = []
    for i, aspect_ratio in enumerate(aspect_ratio_array):
        bs = load(f"./biot_savart_aspect_{aspect_ratio}.json")

        # compute B with taylor expansion
        B_taylor = stel.B_taylor(r=stel.rc[0].detach().numpy()/aspect_ratio, ntheta=32).detach().numpy()
        
        # compute B with bs
        points = stel.surface(r=stel.rc[0].detach().numpy()/aspect_ratio, ntheta=32).detach().numpy()
        points = points.reshape((-1,3))
        bs.set_points(np.ascontiguousarray(points))
        B_bs = bs.B().reshape((stel.nphi, -1, 3))
        # normalize
        axis_field = B_taylor/np.mean(np.linalg.norm(B_taylor, axis=-1))
        coil_field = B_bs/np.mean(np.linalg.norm(B_bs, axis=-1))
        # error
        B_error = axis_field - coil_field
        error_array_1.append(np.max(np.abs(B_error)))

    plt.plot(1/np.array(aspect_ratio_array), error_array_1)
    c3 = aspect_ratio_array[0]**3 * error_array_1[0]
    c2 = aspect_ratio_array[0]**2 * error_array_1[0]
    plt.plot(1/np.array(aspect_ratio_array), [c3/ar**3 for ar in aspect_ratio_array], 'r--', label='1/aspect_ratio^3')
    plt.plot(1/np.array(aspect_ratio_array), [c2/ar**2 for ar in aspect_ratio_array], 'k--', label='1/aspect_ratio^2')
    plt.xlabel('inverse aspect ratio')
    plt.ylabel('error [T]')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

def test_B_cross_n():

    error_array_1 = []
    error_array_2 = []
    for i, aspect_ratio in enumerate(aspect_ratio_array):
        bs = load(f"./biot_savart_aspect_{aspect_ratio}.json")

        # compute Bxn with NAE
        r = stel.rc[0].detach().numpy()/aspect_ratio
        dsurface_by_dtheta = stel.dsurface_by_dtheta(r=r, ntheta=32).detach().numpy()
        G = (stel.G0).detach().numpy()
        B_cross_n = G * dsurface_by_dtheta

        # compute Bxn with biot savart
        points = stel.surface(r=r, ntheta=32).detach().numpy()
        points = points.reshape((-1,3))
        bs.set_points(np.ascontiguousarray(points))
        B_bs = bs.B().reshape((stel.nphi, -1, 3))
        normal = stel.surface_normal(r=r, ntheta=32).detach().numpy()
        B_cross_n_bs = np.cross(B_bs, normal, axis=-1)

        # compute with taylor field
        B_taylor = stel.B_taylor(r=r, ntheta=32).detach().numpy()
        B_cross_n_taylor = np.cross(B_taylor, normal, axis=-1)
        
        # normalize
        axis_field = B_cross_n/np.mean(np.linalg.norm(B_cross_n, axis=-1))
        coil_field = B_cross_n_bs/np.mean(np.linalg.norm(B_cross_n_bs, axis=-1))
        taylor_field = B_cross_n_taylor/np.mean(np.linalg.norm(B_cross_n_taylor, axis=-1))
        # error
        B_error = axis_field - coil_field
        error_array_1.append(np.max(np.abs(B_error)))
        B_error = taylor_field - axis_field
        error_array_2.append(np.max(np.abs(B_error)))

    # plt.plot(1/np.array(aspect_ratio_array), error_array_1, markersize=8, marker='o')
    plt.plot(1/np.array(aspect_ratio_array), error_array_2, markersize=8, marker='s')
    c3 = aspect_ratio_array[0]**3 * error_array_2[0]
    c2 = aspect_ratio_array[0]**2 * error_array_2[0]
    c1 = aspect_ratio_array[0]**1 * error_array_2[0]
    plt.plot(1/np.array(aspect_ratio_array), [c1/ar**1 for ar in aspect_ratio_array], 'g--', label='1/aspect_ratio^1')
    plt.plot(1/np.array(aspect_ratio_array), [c3/ar**3 for ar in aspect_ratio_array], 'r--', label='1/aspect_ratio^3')
    plt.plot(1/np.array(aspect_ratio_array), [c2/ar**2 for ar in aspect_ratio_array], 'k--', label='1/aspect_ratio^2')
    plt.xlabel('inverse aspect ratio')
    plt.ylabel('error [T]')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

def plot_B_cross_n():

    error_array_1 = []
    error_array_2 = []
    for i, aspect_ratio in enumerate(aspect_ratio_array):
        bs = load(f"./biot_savart_aspect_{aspect_ratio}.json")

        # compute Bxn with NAE
        r = stel.rc[0].detach().numpy()/aspect_ratio
        dsurface_by_dtheta = stel.dsurface_by_dtheta(r=r, ntheta=32).detach().numpy()
        G = (stel.G0).detach().numpy()
        B_cross_n = G * dsurface_by_dtheta

        # compute Bxn with biot savart
        points = stel.surface(r=r, ntheta=32).detach().numpy()
        points = points.reshape((-1,3))
        normal = stel.surface_normal(r=r, ntheta=32).detach().numpy()

        # compute with taylor field
        B_taylor = stel.B_taylor(r=r, ntheta=32).detach().numpy()
        B_cross_n_taylor = np.cross(B_taylor, normal, axis=-1)
        
        # B_error = np.mean(np.linalg.norm(B_cross_n,axis=-1))
        # error_array_2.append(B_error)
        # B_error = np.mean(np.linalg.norm(B_cross_n_taylor,axis=-1))
        # error_array_1.append(B_error)

        # compute Bxn with biot savart
        points = stel.surface(r=r, ntheta=32).detach().numpy()
        points = points.reshape((-1,3))
        bs.set_points(np.ascontiguousarray(points))
        B_bs = bs.B().reshape((stel.nphi, -1, 3))
        normal = stel.surface_normal(r=r, ntheta=32).detach().numpy()
        B_cross_n_bs = np.cross(B_bs, normal, axis=-1)
        B_cross_n_bs = np.mean(np.linalg.norm(B_cross_n, axis=-1)) * B_cross_n_bs / np.mean(np.linalg.norm(B_cross_n_bs, axis=-1))

        B_error = np.mean(np.linalg.norm(B_cross_n - B_cross_n_bs,axis=-1))
        error_array_1.append(B_error)

        B_error = np.mean(np.linalg.norm(B_cross_n - B_cross_n_taylor,axis=-1))
        error_array_2.append(B_error)

    plt.plot(1/np.array(aspect_ratio_array), error_array_1, markersize=8, marker='o', label='BS err')
    plt.plot(1/np.array(aspect_ratio_array), error_array_2, markersize=8, marker='s', label='B_taylor err')
    c3 = aspect_ratio_array[0]**3 * error_array_2[0]
    c2 = aspect_ratio_array[0]**2 * error_array_2[0]
    c1 = aspect_ratio_array[0]**1 * error_array_2[0]
    plt.plot(1/np.array(aspect_ratio_array), [c1/ar**1 for ar in aspect_ratio_array], 'g--', label='1/aspect_ratio^1')
    plt.plot(1/np.array(aspect_ratio_array), [c2/ar**2 for ar in aspect_ratio_array], 'k--', label='1/aspect_ratio^2')
    plt.plot(1/np.array(aspect_ratio_array), [c3/ar**3 for ar in aspect_ratio_array], 'r--', label='1/aspect_ratio^3')
    plt.xlabel('inverse aspect ratio')
    plt.ylabel('error [T]')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

def plot_virtual_casing_integrand():

    xyz0 = stel.XYZ0.T.detach().numpy() # (nphi,3)

    error_array_1 = []
    error_array_2 = []
    for i, aspect_ratio in enumerate(aspect_ratio_array):
        bs = load(f"./biot_savart_aspect_{aspect_ratio}.json")

        # compute Bxn with NAE
        r = stel.rc[0].detach().numpy()/aspect_ratio
        dsurface_by_dtheta = stel.dsurface_by_dtheta(r=r, ntheta=32).detach().numpy()
        G = (stel.G0).detach().numpy()
        B_cross_n = G * dsurface_by_dtheta

        # compute Bxn with biot savart
        points = stel.surface(r=r, ntheta=32).detach().numpy()
        points = points.reshape((-1,3))
        normal = stel.surface_normal(r=r, ntheta=32).detach().numpy()

        # compute with taylor field
        B_taylor = stel.B_taylor(r=r, ntheta=32).detach().numpy()
        B_cross_n_taylor = np.cross(B_taylor, normal, axis=-1)

        # compute Bxn with biot savart
        bs.set_points(np.ascontiguousarray(points))
        B_bs = bs.B().reshape((stel.nphi, -1, 3))
        normal = stel.surface_normal(r=r, ntheta=32).detach().numpy()
        B_cross_n_bs = np.cross(B_bs, normal, axis=-1)
        B_cross_n_bs = np.mean(np.linalg.norm(B_cross_n, axis=-1)) * B_cross_n_bs / np.mean(np.linalg.norm(B_cross_n_bs, axis=-1))

        # cross with kernel
        error_array_1_phi = np.zeros(stel.nphi)
        error_array_2_phi = np.zeros(stel.nphi)
        for idx in range(stel.nphi):
            delta_r = points.reshape((stel.nphi, -1, 3)) - xyz0[idx]
            kernel = delta_r / (np.linalg.norm(delta_r,axis=-1, keepdims=True)**3)
            k_cross_B_cross_n = np.linalg.cross(kernel, B_cross_n, axis=-1)
            k_cross_B_cross_n_taylor = np.linalg.cross(kernel, B_cross_n_taylor, axis=-1)
            k_cross_B_cross_n_bs = np.linalg.cross(kernel, B_cross_n_bs, axis=-1)

            # change of variables
            d_varphi_d_phi = stel.d_varphi_d_phi.detach().numpy().reshape((-1,1,1))
            k_cross_B_cross_n = k_cross_B_cross_n * d_varphi_d_phi
            k_cross_B_cross_n_taylor = k_cross_B_cross_n_taylor * d_varphi_d_phi
            k_cross_B_cross_n_bs = k_cross_B_cross_n_bs * d_varphi_d_phi

            # use mean error to simulate integration over surface
            B_error = np.mean(np.linalg.norm(k_cross_B_cross_n - k_cross_B_cross_n_bs,axis=-1))
            error_array_1_phi[idx] = B_error

            B_error = np.mean(np.linalg.norm(k_cross_B_cross_n - k_cross_B_cross_n_taylor,axis=-1))
            error_array_2_phi[idx] = B_error

        # worst case error over axis
        error_array_1.append(np.max(error_array_1_phi))
        error_array_2.append(np.max(error_array_2_phi))

    plt.plot(1/np.array(aspect_ratio_array), error_array_1, markersize=8, marker='o', label='BS err')
    plt.plot(1/np.array(aspect_ratio_array), error_array_2, markersize=8, marker='s', label='B_taylor err')
    c3 = aspect_ratio_array[0]**3 * error_array_2[0]
    c2 = aspect_ratio_array[0]**2 * error_array_2[0]
    c1 = aspect_ratio_array[0]**1 * error_array_2[0]
    plt.plot(1/np.array(aspect_ratio_array), [c1/ar**1 for ar in aspect_ratio_array], 'g--', label='1/aspect_ratio^1')
    plt.plot(1/np.array(aspect_ratio_array), [c2/ar**2 for ar in aspect_ratio_array], 'k--', label='1/aspect_ratio^2')
    plt.plot(1/np.array(aspect_ratio_array), [c3/ar**3 for ar in aspect_ratio_array], 'r--', label='1/aspect_ratio^3')
    plt.xlabel('inverse aspect ratio')
    plt.ylabel('error [T]')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.show()


def plot_virtual_casing_integral():

    ntheta = 32
    dtheta = 2 * np.pi / ntheta
    dphi = np.diff(stel.phi.detach().numpy())[0]

    xyz0 = stel.XYZ0.T.detach().numpy() # (nphi,3)

    error_array_1 = []
    error_array_2 = []
    for i, aspect_ratio in enumerate(aspect_ratio_array):
        bs = load(f"./biot_savart_aspect_{aspect_ratio}.json")

        # compute Bxn with NAE
        r = stel.rc[0].detach().numpy()/aspect_ratio
        dsurface_by_dtheta = stel.dsurface_by_dtheta(r=r, ntheta=ntheta).detach().numpy()
        G = (stel.G0).detach().numpy()
        B_cross_n = G * dsurface_by_dtheta

        # compute Bxn with biot savart
        points = stel.surface(r=r, ntheta=ntheta).detach().numpy()
        points = points.reshape((-1,3))
        normal = stel.surface_normal(r=r, ntheta=ntheta).detach().numpy()

        # compute with taylor field
        B_taylor = stel.B_taylor(r=r, ntheta=ntheta).detach().numpy()
        B_cross_n_taylor = np.cross(B_taylor, normal, axis=-1)

        # compute Bxn with biot savart
        bs.set_points(np.ascontiguousarray(points))
        B_bs = bs.B().reshape((stel.nphi, -1, 3))
        normal = stel.surface_normal(r=r, ntheta=ntheta).detach().numpy()
        B_cross_n_bs = np.cross(B_bs, normal, axis=-1)
        B_cross_n_bs = np.mean(np.linalg.norm(B_cross_n, axis=-1)) * B_cross_n_bs / np.mean(np.linalg.norm(B_cross_n_bs, axis=-1))

        # multiply by area element
        d_varphi_d_phi = stel.d_varphi_d_phi.detach().numpy().reshape((-1,1,1))
        dA = np.tile(d_varphi_d_phi * dphi * dtheta, stel.nfp).reshape((-1,1,1))
        # B_cross_n = B_cross_n * dA
        # B_cross_n_taylor = B_cross_n_taylor * dA
        # B_cross_n_bs = B_cross_n_bs * dA

        # 1nfp rotation matrix
        angle = 2 * np.pi / stel.nfp
        Q = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]]
                        )
        
        # build n x B on full torus
        points = points.reshape((stel.nphi, -1, 3))
        nB = - np.copy(B_cross_n)
        nB_bs = - np.copy(B_cross_n_bs)
        nB_taylor = - np.copy(B_cross_n_taylor)
        n_cross_B = np.zeros((int(stel.nfp * stel.nphi), ntheta, 3))
        n_cross_B_bs = np.zeros((int(stel.nfp * stel.nphi), ntheta, 3))
        n_cross_B_taylor = np.zeros((int(stel.nfp * stel.nphi), ntheta, 3))
        surface = np.zeros((int(stel.nfp * stel.nphi), ntheta, 3))
        for ii in range(stel.nfp):
            points = np.einsum('ij,klj->kli', Q, points)
            nB = np.einsum('ij,klj->kli', Q, nB)
            nB_bs = np.einsum('ij,klj->kli', Q, nB_bs)
            nB_taylor = np.einsum('ij,klj->kli', Q, nB_taylor)
            surface[ii * stel.nphi : (ii+1) * stel.nphi] = np.copy(points)
            n_cross_B[ii * stel.nphi : (ii+1) * stel.nphi] = np.copy(nB)
            n_cross_B_bs[ii * stel.nphi : (ii+1) * stel.nphi] = np.copy(nB_bs)
            n_cross_B_taylor[ii * stel.nphi : (ii+1) * stel.nphi] = np.copy(nB_taylor)

        # # TODO:remove
        # points = np.einsum('ij,klj->kli', Q, points)
        # B_cross_n = np.einsum('ij,klj->kli', Q, B_cross_n)
        # B_cross_n_taylor = np.einsum('ij,klj->kli', Q, B_cross_n_taylor)
        # B_cross_n_bs = np.einsum('ij,klj->kli', Q, B_cross_n_bs)

        # cross with kernel
        error_array_1_phi = np.zeros(stel.nphi)
        error_array_2_phi = np.zeros(stel.nphi)
        for idx in range(stel.nphi):
        # for idx in range(48,50):

            # TODO:uncomment
            delta_r = surface - xyz0[idx]
            kernel = delta_r / (np.linalg.norm(delta_r,axis=-1, keepdims=True)**3)
            k_cross_n_cross_B = np.linalg.cross(kernel, n_cross_B, axis=-1)
            k_cross_n_cross_B_taylor = np.linalg.cross(kernel, n_cross_B_taylor, axis=-1)
            k_cross_n_cross_B_bs = np.linalg.cross(kernel, n_cross_B_bs, axis=-1)

            # # # TODO: remove
            # # use mean error to simulate integration over surface
            # B_error = np.mean(np.linalg.norm(k_cross_n_cross_B - k_cross_n_cross_B_bs, axis=-1))
            # error_array_1_phi[idx] = B_error
            # B_error = np.mean(np.linalg.norm(k_cross_n_cross_B - k_cross_n_cross_B_taylor, axis=-1))
            # error_array_2_phi[idx] = B_error

            # # TODO: remove
            # delta_r = points - xyz0[idx]
            # kernel = delta_r / (np.linalg.norm(delta_r,axis=-1, keepdims=True)**3)
            # k_cross_B_cross_n = np.linalg.cross(kernel, B_cross_n, axis=-1)
            # k_cross_B_cross_n_taylor = np.linalg.cross(kernel, B_cross_n_taylor, axis=-1)
            # k_cross_B_cross_n_bs = np.linalg.cross(kernel, B_cross_n_bs, axis=-1)

            # # TODO: remove
            # # sanity check plot
            # ax = plt.figure().add_subplot(projection='3d')
            # ax.plot_wireframe(points[:,:,0], points[:,:,1], points[:,:,2], alpha=0.2) # ours
            # # points = np.einsum('ij,klj->kli', Q, points)
            # # ax.plot_wireframe(points[:,:,0], points[:,:,1], points[:,:,2], color='r', alpha=0.2) # ours
            # ax.scatter([xyz0[idx,0]],[xyz0[idx,1]], [xyz0[idx,2]], color='k', alpha=0.2) # ours
            # plt.show()
            # quit()

            # # TODO: remove
            # # k x n x B dA / 4pi
            # dA = (d_varphi_d_phi * dphi * dtheta).reshape((-1,1,1))
            # integrand = k_cross_B_cross_n * dA / 4 / np.pi
            # integrand_bs = k_cross_B_cross_n_bs * dA / 4 / np.pi
            # integrand_taylor = k_cross_B_cross_n_taylor * dA / 4 / np.pi

            # # TODO: remove
            # B_error = np.max(np.linalg.norm(k_cross_B_cross_n - k_cross_B_cross_n_bs,axis=-1))
            # error_array_1_phi[idx] = B_error
            # B_error = np.max(np.linalg.norm(k_cross_B_cross_n - k_cross_B_cross_n_taylor,axis=-1))
            # error_array_2_phi[idx] = B_error

            # TODO: uncomment
            # k x n x B dA / 4pi
            integrand = k_cross_n_cross_B * dA / 4 / np.pi
            integrand_bs = k_cross_n_cross_B_bs * dA / 4 / np.pi
            integrand_taylor = k_cross_n_cross_B_taylor * dA / 4 / np.pi

            # # # TODO: remove
            # # use mean error to simulate integration over surface
            # B_error = np.mean(np.linalg.norm(integrand - integrand_bs, axis=-1))
            # error_array_1_phi[idx] = B_error
            # B_error = np.mean(np.linalg.norm(integrand - integrand_taylor, axis=-1))
            # error_array_2_phi[idx] = B_error
            
            # TODO: uncomment
            # integral
            total = np.sum(integrand, axis=(0,1))
            total_bs = np.sum(integrand_bs, axis=(0,1))
            total_taylor = np.sum(integrand_taylor, axis=(0,1))

            # TODO: uncomment
            # use mean error to simulate integration over surface
            B_error = np.linalg.norm(total - total_bs)
            error_array_1_phi[idx] = B_error
            B_error = np.linalg.norm(total - total_taylor)
            error_array_2_phi[idx] = B_error

        # worst case error over axis
        error_array_1.append(np.max(error_array_1_phi))
        error_array_2.append(np.max(error_array_2_phi))

    plt.plot(1/np.array(aspect_ratio_array), error_array_1, markersize=8, marker='o', label='BS err')
    plt.plot(1/np.array(aspect_ratio_array), error_array_2, markersize=8, marker='s', label='B_taylor err')
    c4 = aspect_ratio_array[0]**4 * error_array_2[0]
    c3 = aspect_ratio_array[0]**3 * error_array_2[0]
    c2 = aspect_ratio_array[0]**2 * error_array_2[0]
    c1 = aspect_ratio_array[0]**1 * error_array_2[0]
    plt.plot(1/np.array(aspect_ratio_array), [c1/ar**1 for ar in aspect_ratio_array], 'g--', label='1/aspect_ratio^1')
    plt.plot(1/np.array(aspect_ratio_array), [c2/ar**2 for ar in aspect_ratio_array], 'k--', label='1/aspect_ratio^2')
    plt.plot(1/np.array(aspect_ratio_array), [c3/ar**3 for ar in aspect_ratio_array], 'r--', label='1/aspect_ratio^3')
    plt.plot(1/np.array(aspect_ratio_array), [c4/ar**4 for ar in aspect_ratio_array], '--', color='magenta', label='1/aspect_ratio^4')
    plt.xlabel('inverse aspect ratio')
    plt.ylabel('error [T]')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.show()





if __name__=="__main__":
    # for aspect_ratio in aspect_ratio_array:
    #     optimize_coils(aspect_ratio)
    # test_coils()
    # test_B_on_surface()
    # plot_B_cross_n()
    # plot_virtual_casing_integrand()
    plot_virtual_casing_integral()