import numpy as np
from simsopt.geo import create_equally_spaced_curves
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.geo import CurveLength, LpCurveCurvature, CurveCurveDistance, MeanSquaredCurvature
from simsopt.objectives import QuadraticPenalty
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from simsopt.geo import plot as sms_plot
from qsc.simsopt_objectives import (QscOptimizable, ExternalFieldError,
                                    GradExternalFieldError)

"""
Stage 2 optimization example.
"""

# configuration parameters
ncoils = 4
nfp = 2
is_stellsym = True
coil_major_radius = 1.0
coil_minor_radius = 0.5
coil_n_fourier_modes = 3
coil_current = 100000.0

# last closed flux surface
aspect_ratio = 10.0

# coil constraint parameters
coil_length_target = 3.0 # length of each coil
coil_curvature_target = 5.0 
cc_dist_target = 0.2 # coil separation

# optimization parameters
max_iter = 100
mu_penalty = 1.0

""" initialization """

# initialize axis
stel = QscOptimizable.from_paper("precise QA", p2=-1e5)
nfp = stel.nfp
minor_radius = stel.rc[0].detach().numpy().item() / aspect_ratio

# initialize coils
base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=is_stellsym, R0=coil_major_radius,
                                        R1=coil_minor_radius, order=coil_n_fourier_modes)
base_currents = [Current(1.0) * coil_current for i in range(ncoils)]
coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=is_stellsym)
biot_savart = BiotSavart(coils)


# plot the coils and axis
xyz0 = stel.XYZ0.detach().numpy() # (3, nphi)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(xyz0[0], xyz0[1], xyz0[2])
sms_plot(coils, engine="matplotlib", ax=ax, close=True, show=False)
surface = stel.surface(r=minor_radius, ntheta=64).detach().numpy()
ax.plot_surface(surface[:,:,0], surface[:,:,1], surface[:,:,2], alpha=0.3, color='lightgray')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
plt.show()


""" set up the optimization problem """

# choose dofs
stel.fix_all()
biot_savart.unfix_all()

# field matching objectives
ntheta_vc = 256
nphi_vc = 1024
ntarget = stel.nphi
fe = ExternalFieldError(biot_savart, stel, r=minor_radius, ntheta=ntheta_vc, nphi=nphi_vc, ntarget=ntarget)
ge = GradExternalFieldError(biot_savart, stel, r=minor_radius, ntheta=ntheta_vc, nphi=nphi_vc, ntarget=ntarget)

# coil constraints/regularization
coil_lengths_penalties = [(1 / coil_length_target**2) * QuadraticPenalty(CurveLength(c), coil_length_target, "identity") for c in base_curves]
coil_curvature_penalties = [(1 / coil_curvature_target**2) * LpCurveCurvature(c, 2, threshold=coil_curvature_target) for c in base_curves]
msc_penalties = [(1 / coil_curvature_target**2) * QuadraticPenalty(MeanSquaredCurvature(c), coil_curvature_target, "max") for c in base_curves]
cc_dist_penalty = CurveCurveDistance([c.curve for c in coils], cc_dist_target)

# form an Optimizable objective
prob = (fe 
        + ge
        + mu_penalty * sum(coil_lengths_penalties) 
        + mu_penalty * sum(coil_curvature_penalties)
        + mu_penalty * sum(msc_penalties)
        + mu_penalty * cc_dist_penalty
        )
def fun(dofs):
    biot_savart.x = dofs
    return prob.J(), prob.dJ()


""" solve the optimization problem """

print("\nInitial results")
print('field error', fe.J())
print('grad field error', ge.J())
coil_lengths = [CurveLength(c).J() for c in base_curves]
print('coil length min, max', np.min(coil_lengths), np.max(coil_lengths))
print('min curve separation', cc_dist_penalty.shortest_distance())
curvatures = [MeanSquaredCurvature(c).J() for c in base_curves]
print('mean squared curvatures min, max', np.min(curvatures), np.max(curvatures))
curvatures = [np.max(c.kappa()) for c in base_curves]
print('max curvatures', np.max(curvatures))

def callback(intermediate_result):
    print("J = ", intermediate_result.fun)

x0 = biot_savart.x
res = minimize(fun, x0=x0, jac=True, method="BFGS", callback=callback, 
                  options={"maxiter":max_iter, "gtol":1e-8})
biot_savart.x = res.x

# evaluate the solution
print("\nOptimized results")
print('total field error', fe.J())
print('grad field error', ge.J())
coil_lengths = [CurveLength(c).J() for c in base_curves]
print('coil length min, max', np.min(coil_lengths), np.max(coil_lengths))
print('min curve separation', cc_dist_penalty.shortest_distance())
curvatures = [MeanSquaredCurvature(c).J() for c in base_curves]
print('mean squared curvatures min, max', np.min(curvatures), np.max(curvatures))
curvatures = [np.max(c.kappa()) for c in base_curves]
print('max curvatures', np.max(curvatures))


# plot the coils and axis
xyz0 = stel.XYZ0.detach().numpy() # (3, nphi)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(xyz0[0], xyz0[1], xyz0[2])
sms_plot(coils, engine="matplotlib", ax=ax, close=True, show=False)
surface = stel.surface(r=minor_radius, ntheta=ntheta_vc).detach().numpy()
ax.plot_surface(surface[:,:,0], surface[:,:,1], surface[:,:,2], alpha=0.3, color='lightgray')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
plt.show()