import numpy as np
import os
import pickle
from qsc.qsc import Qsc
import torch
import time
from simsopt.geo import create_equally_spaced_curves
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.geo import CurveLength, LpCurveCurvature, LpCurveTorsion
from simsopt.objectives import QuadraticPenalty
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from simsopt.geo import plot as sms_plot
from qsc.simsopt_objectives import (QscOptimizable, FieldError, ExternalFieldError,
                                    IotaPenalty, AxisLengthPenalty, GradExternalFieldError,
                                    GradBPenalty, GradGradBPenalty, B20Penalty, MagneticWellPenalty,
                                    PressurePenalty)

# configuration parameters
ncoils = 4
nfp = 2
is_stellsym = True
coil_major_radius = 1.0
coil_minor_radius = 0.5
coil_n_fourier_modes = 3
coil_current = 100000.0 

# axis parameters
order = 'r3'
axis_n_fourier_modes = 7
etabar = 1.0
axis_nphi = 31

# B_external computation
minor_radius = 0.1
ntheta_vc = 64
nphi_vc = 256


# constraints
iota_target = 0.61 # target iota
coil_length_weight = 1.0 # weight on coil length penalty
coil_length_target = 3.0 # length of each coil
axis_length_target = 6.28
coil_curvature_target = 5.0

# optimization parameters
max_iter = 100
mu_penalty = 1.0

""" initialization """

# initialize coils
base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=is_stellsym, R0=coil_major_radius,
                                        R1=coil_minor_radius, order=coil_n_fourier_modes)
base_currents = [Current(1.0) * coil_current for i in range(ncoils)]
coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=is_stellsym)
biot_savart = BiotSavart(coils)

# set up the expansion
rc = np.zeros(axis_n_fourier_modes)
zs = np.zeros(axis_n_fourier_modes)

# initialization
rc[0] = axis_length_target / (2 * np.pi)
zs[1] = 1e-4

stel = QscOptimizable(rc=rc, zs=zs, etabar=etabar, order=order, nphi=axis_nphi, nfp=nfp)

# choose degrees of freedom
stel.fix_all()
stel.unfix('etabar')
for ii in range(1, axis_n_fourier_modes):
    stel.unfix(f'rc({ii})')
    stel.unfix(f'zs({ii})')
stel.unfix('B2c')

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


""" set up the optimization problem """

# field matching objectives
fe = ExternalFieldError(biot_savart, stel, r=minor_radius, ntheta=ntheta_vc, nphi=nphi_vc)
ge = GradExternalFieldError(biot_savart, stel, r=minor_radius, ntheta=ntheta_vc, nphi=nphi_vc)

# quasi-symmetry
b20_penalty = B20Penalty(stel)

# constraints/regularization
iota_penalty = IotaPenalty(stel, iota_target)
coil_lengths_penalties = [(1 / coil_length_target**2) * QuadraticPenalty(CurveLength(c), coil_length_target, "identity") for c in base_curves]
coil_curvature_penalties = [(1 / coil_curvature_target**2) * LpCurveCurvature(c, 2, threshold=coil_curvature_target) for c in base_curves]
axis_length_penalty = AxisLengthPenalty(stel, axis_length_target)
gradb_penalty = GradBPenalty(stel)
gradgradb_penalty = GradGradBPenalty(stel)
p2_penalty = PressurePenalty(stel, -1e6)

# form an Optimizable objective
prob = (fe 
        + ge
        + mu_penalty * iota_penalty 
        + mu_penalty * sum(coil_lengths_penalties) 
        # + mu_penalty * axis_length_penalty
        + mu_penalty * sum(coil_curvature_penalties)
        + 0.1 * gradb_penalty
        + 0.01 * gradgradb_penalty
        )
def fun(dofs):
    prob.x = dofs
    return prob.J(), prob.dJ()


""" solve the optimization problem """

print("\nInitial results")
print('field error', fe.J())
print('grad field error', ge.J())
print('iota', stel.iota)
print('axis length', stel.axis_length)
coil_lengths = [CurveLength(c).J() for c in base_curves]
print('coil length min, max', np.min(coil_lengths), np.max(coil_lengths))

def callback(intermediate_result):
    print("J = ", intermediate_result.fun)

x0 = prob.x
res = minimize(fun, x0=x0, jac=True, method="BFGS", callback=callback, 
                  options={"maxiter":max_iter, "gtol":1e-8})
prob.x = res.x

# evaluate the solution
xopt = prob.x
print("\nOptimized results")
print('total field error', fe.J())
print('grad field error', ge.J())
print('iota', stel.iota)
print('axis length', stel.axis_length)
coil_lengths = [CurveLength(c).J() for c in base_curves]
print('coil length min, max', np.min(coil_lengths), np.max(coil_lengths))


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


""" Now that we have a good warm start, optimize with pressure"""

max_iter = 50
stel.unfix('B2c')
stel.unfix('I2')
stel.unfix('p2')

# form an Optimizable objective
prob = (fe 
        + ge 
        + mu_penalty * iota_penalty 
        + mu_penalty * sum(coil_lengths_penalties) 
        + mu_penalty * axis_length_penalty
        + mu_penalty * sum(coil_curvature_penalties)
        + 0.1 * gradb_penalty
        + 0.01 * gradgradb_penalty
        + p2_penalty
        )
def fun(dofs):
    prob.x = from_unit_scale(dofs)
    grad = chain_rule(prob.dJ())
    return prob.J(), grad
def to_unit_scale(dofs):
    """scale p2 to [0,1]"""
    dofs[-2] = dofs[-2] / 1e6
    return dofs
def from_unit_scale(dofs):
    """scale p2 to [0,1e6]"""
    dofs[-2] = dofs[-2] * 1e6
    return dofs
def chain_rule(J):
    """compute gradient with respect to unit scale variables"""
    J[-2] = J[-2] * 1e6
    return J

x0 = to_unit_scale(prob.x)
res = minimize(fun, x0=x0, jac=True, method="BFGS", callback=callback, 
                  options={"maxiter":max_iter, "gtol":1e-8})

prob.x = from_unit_scale(res.x)

# evaluate the solution
print("\nOptimized results")
print('total field error', fe.J())
print('grad field error', ge.J())
print('iota', stel.iota)
print('pressure', stel.p2)
print('current', stel.I2)
print('axis length', stel.axis_length)
coil_lengths = [CurveLength(c).J() for c in base_curves]
print('coil length min, max', np.min(coil_lengths), np.max(coil_lengths))
print(stel.rc)
print(stel.zs)

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