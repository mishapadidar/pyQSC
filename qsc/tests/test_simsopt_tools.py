import matplotlib.pyplot as plt
from simsopt.geo import plot
from simsopt.field import Current, coils_via_symmetries
from qsc.qsc import Qsc
from qsc.simsopt_tools import create_equally_spaced_curves_around_axis


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

if __name__ == "__main__":
    test_create_equally_spaced_curves_around_axis()