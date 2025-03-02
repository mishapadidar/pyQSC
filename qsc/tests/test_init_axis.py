import numpy as np
from qsc import Qsc
import torch
import matplotlib.pyplot as plt
from simsopt.geo import CurveRZFourier


def test_init_axis():
    stel = Qsc.from_paper("precise QA", nphi=7, order='r2')
    curve = CurveRZFourier(quadpoints=stel.phi.detach().numpy()/2/np.pi, order = stel.nfourier-1, nfp=stel.nfp, stellsym=True)
    curve.unfix_all()
    curve.x = torch.concatenate((stel.rc, stel.zs[1:])).flatten().detach().numpy()
    
    # check axis shape
    axis_qsc = stel.XYZ0.detach().numpy().T # (nphi, 3)
    axis_sm =curve.gamma()
    err = axis_qsc - axis_sm
    assert np.max(np.abs(err)) < 1e-14, "XYZ0 is incorrect."

    # ax = plt.figure().add_subplot(projection='3d')
    # plt.plot(axis_qsc[:,0], axis_qsc[:,1], axis_qsc[:,2], label='qsc')
    # plt.plot(axis_sm[:,0], axis_sm[:,1], axis_sm[:,2])
    # plt.legend()
    # plt.show()

    # check axis derivative
    err = stel.dXYZ0_by_dphi.detach().numpy().T - curve.gammadash()/(2*np.pi)
    assert np.max(np.abs(err)) < 1e-14, "dXYZ0_by_dphi is incorrect."

    # check frenet-frame
    (t,n,b) = curve.frenet_frame()
    err = stel.tangent_cartesian.detach().numpy() - t
    assert np.max(np.abs(err)) < 1e-14, "tangent_cartesian is incorrect."

    err = stel.normal_cartesian.detach().numpy() - n
    assert np.max(np.abs(err)) < 1e-14, "normal_cartesian is incorrect."

    err = stel.binormal_cartesian.detach().numpy() - b
    assert np.max(np.abs(err)) < 1e-14, "binormal_cartesian is incorrect."


if __name__ == "__main__":
    test_init_axis()
