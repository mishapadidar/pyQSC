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
                                    LGradB, B20Penalty, MagneticWellPenalty)

# configuration parameters
ncoils = 4
nfp = 2
is_stellsym = True
coil_major_radius = 1.0
coil_minor_radius = 0.5
coil_n_fourier_modes = 3
coil_current = 100000.0 

# axis parameters
order = 'r2'
axis_n_fourier_modes = 5
etabar = 1.0
axis_nphi = 31

# B_external computation
minor_radius = 0.1
ntheta_vc = 256
nphi_vc = 1024
ntarget = axis_nphi

# constraints
iota_target = 0.103 # target iota
coil_length_weight = 1.0 # weight on coil length penalty
coil_length_target = 4.398 # length of each coil
axis_length_target = 6.28
coil_curvature_target = 2 * 2 * np.pi / coil_length_target
well_target = -50

# optimization parameters
max_iter = 30
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

# optional warm start: first order solution
data = pickle.load(open("./output/coil_opt_data.pickle","rb"))
stel.x = data['axis']
biot_savart.x = data['bs']

# choose degrees of freedom
stel.fix_all()
stel.unfix('etabar')
for ii in range(1, axis_n_fourier_modes):
    stel.unfix(f'rc({ii})')
    stel.unfix(f'zs({ii})')
# fix pressure profile
stel.fix('p2')

# plot the coils and axis
xyz0 = stel.XYZ0.detach().numpy() # (3, nphi)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(xyz0[0], xyz0[1], xyz0[2])
sms_plot(coils, engine="matplotlib", ax=ax, close=True, show=False)
plt.show()

""" set up the optimization problem """

# field matching objectives
fe = ExternalFieldError(biot_savart, stel, r=minor_radius, ntheta=ntheta_vc, nphi=nphi_vc, ntarget=ntarget)
ge = GradExternalFieldError(biot_savart, stel, r=minor_radius, ntheta=ntheta_vc, nphi=nphi_vc, ntarget=ntarget)

# quasi-symmetry
b20_penalty = B20Penalty(stel)

# constraints/regularization
iota_penalty = IotaPenalty(stel, iota_target)
coil_lengths_penalties = [(1 / coil_length_target**2) * QuadraticPenalty(CurveLength(c), coil_length_target, "identity") for c in base_curves]
coil_curvature_penalties = [(1 / coil_curvature_target**2) * LpCurveCurvature(c, 2, threshold=coil_curvature_target) for c in base_curves]
axis_length_penalty = AxisLengthPenalty(stel, axis_length_target)
lgradb_penalty = LGradB(stel)
well_penalty = MagneticWellPenalty(stel, well_target=well_target)

# form an Optimizable objective
constraint_violation = (iota_penalty + sum(coil_lengths_penalties) + axis_length_penalty + sum(coil_curvature_penalties))
optional_penalties = lgradb_penalty + well_penalty
prob = fe + ge + mu_penalty * constraint_violation #+ mu_penalty * b20_penalty + mu_penalty * optional_penalties 
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

t0 = time.time()
# res = minimize(fun, x0=prob.x, jac=True, method="L-BFGS-B", options={"maxiter":max_iter, "iprint":5})
res = minimize(fun, x0=prob.x, jac=True, method="BFGS", callback=callback, 
                  options={"maxiter":max_iter, "gtol":1e-8})
t1 = time.time()
print("\nIteration time:", (t1-t0)/max_iter)

prob.x = res.x

# evaluate the solution
xopt = prob.x
print("\nOptimized results")
print('total field error', fe.J())
print('iota', stel.iota)
print('axis length', stel.axis_length)
coil_lengths = [CurveLength(c).J() for c in base_curves]
print('coil length min, max', np.min(coil_lengths), np.max(coil_lengths))


""" save results """

outdir = "./output"
outfilename = outdir + "/coil_opt_data.pickle"
print("\nSaving data to:", outfilename)
os.makedirs(outdir, exist_ok=True)
prob.unfix_all()
data = {'axis': stel.x, 'bs':biot_savart.x}
pickle.dump(data, open(outfilename,"wb"))

# get axis shape
xyz0 = stel.XYZ0.detach().numpy() # (3, nphi)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(xyz0[0], xyz0[1], xyz0[2])
sms_plot(coils, engine="matplotlib", ax=ax, close=True, show=False)
outfilename = outdir+"/plot.pdf"
print("Saving plot to:", outfilename)
plt.savefig(fname=outfilename, format='pdf')
# plt.show()