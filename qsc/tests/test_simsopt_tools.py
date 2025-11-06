import matplotlib.pyplot as plt
from simsopt.geo import plot
from simsopt.field import Current, coils_via_symmetries
from qsc.qsc import Qsc
from qsc.simsopt_tools import create_equally_spaced_curves_around_axis, to_CurveRZFourier
import numpy as np

def test_create_equally_spaced_curves_around_axis():
    """Test the create_equally_spaced_curves_around_axis function.
    """
    stel = Qsc.from_paper("precise QA")
    ncurves = 4
    curves = create_equally_spaced_curves_around_axis(stel, ncurves, True, R1=0.5, order=6, numquadpoints=64)

    base_currents = [Current(1.0) for i in range(ncurves)]
    coils = coils_via_symmetries(curves, base_currents, stel.nfp, stellsym=True)

    xyz0 = stel.XYZ0.detach().numpy() # (3, nphi)
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.plot(xyz0[0], xyz0[1], xyz0[2], lw=3)
    plot([c.curve for c in coils], engine="matplotlib", show=False, ax=ax)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    plt.show()

def test_to_CurveRZFourier():
    """Test the to_CurveRZFourier function.
    """
    # check accuracy in QA
    stel = Qsc.from_paper("precise QA")
    curve = to_CurveRZFourier(stel)
    xyz = curve.gamma()
    xyz_actual = stel.XYZ0.detach().numpy().T # (nphi, 3)
    err = np.max(np.abs(xyz - xyz_actual))
    assert err < 1e-15, f"Max error in to_CurveRZFourier is {err}"

    # check accuracy in QH
    stel = Qsc.from_paper("precise QH")
    curve = to_CurveRZFourier(stel)
    xyz = curve.gamma()
    xyz_actual = stel.XYZ0.detach().numpy().T # (nphi, 3)
    err = np.max(np.abs(xyz - xyz_actual))
    assert err < 1e-15, f"Max error in to_CurveRZFourier is {err}"


if __name__ == "__main__":
    test_create_equally_spaced_curves_around_axis()
    test_to_CurveRZFourier()