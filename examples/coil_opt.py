import numpy as np
from qsc.qsc import Qsc
import torch
import numpy as np
from simsopt.geo import create_equally_spaced_curves
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.geo import CurveLength
from simsopt.objectives import QuadraticPenalty
from scipy.optimize import minimize
from qsc.simsopt_objectives import (QscOptimizable, FieldError, ExternalFieldError, IotaPenalty, AxisLengthPenalty)

# configuration parameters
ncoils = 4
nfp = 2
is_stellsym = True
coil_major_radius = 1.0
coil_minor_radius = 0.5
coil_n_fourier_modes = 3
coil_current = 100000.0 

# axis fourier modes
axis_n_fourier_modes = 5
etabar = 1.0
axis_nphi = 511

# constraints
iota_target = 0.103 # target iota
coil_length_weight = 1.0 # weight on coil length penalty
coil_length_target = 4.398 # length of each coil
# coil_length_target = 3.5 # length of each coil
axis_length_target = 6.28

# optimization parameters
max_iter = 800
mu_penalty = 1.0

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

# warm start: first order solution
rc[1:] = [-2.52810494e-05, 1.08331166e-04, 5.40860009e-05, 1.62147715e-02]
zs[1:] = [-2.14779144e-07, -1.92593236e-03, -9.19534498e-06, -7.21549775e-03]
etabar = 1.13558372e+00

stel = QscOptimizable(rc=rc, zs=zs, etabar=etabar, order='r1', nphi=axis_nphi)

# choose degrees of freedom
stel.fix_all()
stel.unfix('etabar')
for ii in range(1, axis_n_fourier_modes):
    stel.unfix(f'rc({ii})')
    stel.unfix(f'zs({ii})')

""" set up the optimization problem """

# physics computations
fe = ExternalFieldError(biot_savart, stel, r=0.3, ntheta=256, ntarget=32)
iota_penalty = IotaPenalty(stel, iota_target)
coil_lengths_penalties = [(1 / coil_length_target**2) * QuadraticPenalty(CurveLength(c), coil_length_target, "identity") for c in base_curves]
axis_length_penalty = AxisLengthPenalty(stel, axis_length_target)
# l_gradB = LGradB(expansion)

# form an Optimizable objective
constraint_violation = (iota_penalty + sum(coil_lengths_penalties) + axis_length_penalty)
prob = fe +  mu_penalty * constraint_violation #+ l_gradB
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

def callback(intermediate_result):
    print(intermediate_result.fun)

# res = minimize(fun, x0=prob.x, jac=True, method="L-BFGS-B", options={"maxiter":max_iter, "iprint":5})
res = minimize(fun, x0=prob.x, jac=True, method="BFGS", callback=callback, 
                  options={"maxiter":max_iter, "gtol":1e-8})

prob.x = res.x

# evaluate the solution
xopt = prob.x
print("\nOptimized results")
print('total field error', fe.J())
print('iota', stel.iota)
print('axis length', stel.axis_length)
coil_lengths = [CurveLength(c).J() for c in base_curves]
print('coil length min, max', np.min(coil_lengths), np.max(coil_lengths))

print('axis dofs', stel.dof_names)
print(stel.x)
