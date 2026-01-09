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
from qsc.simsopt_objectives import (QscOptimizable, FieldError, GradFieldError,
                                    IotaPenalty, AxisLengthPenalty)

# configuration parameters
ncoils = 4
nfp = 2
is_stellsym = True
coil_major_radius = 1.0
coil_minor_radius = 0.7
coil_n_fourier_modes = 10
coil_current = 100000.0 

# axis parameters
order = 'r1'
axis_n_fourier_modes = 6
etabar = 1.0
axis_nphi = 67

# constraints
iota_target = 0.103 # target iota
coil_length_weight = 1.0 # weight on coil length penalty
coil_length_target = 4.398 # length of each coil
axis_length_target = 6.28
coil_curvature_target = 2 * 2 * np.pi / coil_length_target

# optimization parameters
max_iter = 2000
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

stel = QscOptimizable(rc=rc, zs=zs, etabar=etabar, order='r1', nphi=axis_nphi, nfp=nfp)

# choose degrees of freedom
stel.fix_all()
stel.unfix('etabar')
stel.unfix('rc(0)')
for ii in range(1, axis_n_fourier_modes):
    stel.unfix(f'rc({ii})')
    stel.unfix(f'zs({ii})')

# plot the coils and axis
xyz0 = stel.XYZ0.detach().numpy() # (3, nphi)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(xyz0[0], xyz0[1], xyz0[2])
sms_plot(coils, engine="matplotlib", ax=ax, close=True, show=False)
plt.show()

""" set up the optimization problem """

# field matching objectives
fe = FieldError(biot_savart, stel)
ge = GradFieldError(biot_savart, stel)

# constraints/regularization
iota_penalty = IotaPenalty(stel, iota_target)
coil_lengths_penalties = [(0.5 / coil_length_target**2) * QuadraticPenalty(CurveLength(c), coil_length_target, "identity") for c in base_curves]
coil_curvature_penalties = [(1 / coil_curvature_target**2) * LpCurveCurvature(c, 2, threshold=coil_curvature_target) for c in base_curves]
axis_length_penalty = AxisLengthPenalty(stel, axis_length_target)

# form an Optimizable objective
constraint_violation = (iota_penalty + sum(coil_lengths_penalties) + axis_length_penalty + sum(coil_curvature_penalties))
prob = fe + ge + constraint_violation 
def fun(dofs):
    prob.x = dofs
    return prob.J(), prob.dJ()


""" solve the optimization problem """

print("\nInitial results")
print('total field error', fe.J())
print('iota', stel.iota)
print('axis length', stel.axis_length)
coil_lengths = [CurveLength(c).J() for c in base_curves]
print('coil length min, max', np.min(coil_lengths), np.max(coil_lengths))

global i_global
i_global=0

def callback(intermediate_result):
    global i_global
    if i_global % 100 == 0:
        print(f"{i_global}) J = {intermediate_result.fun}", flush=True)
    i_global+= 1

t0 = time.time()
res = minimize(fun, x0=prob.x, jac=True, method="BFGS", callback=callback, 
                  options={"maxiter":max_iter, "gtol":1e-8})
t1 = time.time()
print("\nIteration time:", (t1-t0)/max_iter)

prob.x = res.x

# evaluate the solution
xopt = prob.x
print("\nOptimized results")
print('total field error', fe.J(), flush=True)
print('iota', stel.iota)
print('axis length', stel.axis_length)
coil_lengths = [CurveLength(c).J() for c in base_curves]
print('coil length min, max', np.min(coil_lengths), np.max(coil_lengths))

# plot the coils and axis
xyz0 = stel.XYZ0.detach().numpy() # (3, nphi)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(xyz0[0], xyz0[1], xyz0[2])
sms_plot(coils, engine="matplotlib", ax=ax, close=True, show=False)
plt.show()