import numpy as np
from simsopt.geo import CurveXYZFourier

def create_equally_spaced_curves_around_axis(stel, ncurves, stellsym, R1=0.5, order=6, numquadpoints=64):
    """Initialize circular curves on one field period uniformly spaced around the
    magnetic axis.

    Curve i is initialized around the axis point r0(phi_i) as,
        r_i(theta) = r0(phi_i) + R1 * cos(theta) * n(phi_i) + R1 * sin(theta) * b(phi_i)
    where n, b are the normal and binormal vectors at r0(phi_i). The angles phi_i
    are selected so that the curves are uniformly spaced around the entire torus
    when completed via symmetries.

    Example:
        from simsopt.field import Current, coils_via_symmetries
        from qsc.qsc import Qsc
        stel = Qsc.from_paper("precise QA")
        ncurves = 4
        curves = create_equally_spaced_curves_around_axis(stel, ncurves, True, R1=0.5, order=6, numquadpoints=64)
        base_currents = [Current(1.0) for i in range(ncurves)]
        coils = coils_via_symmetries(curves, base_currents, stel.nfp, stellsym=True)

    Args:
        stel (Qsc): Qsc object
        ncurves (float): number of coils
        stellsym (bool): If True, the curves are initialized on a half field period.
        R1 (float): minor radius of the curves
        order (int): number of Fourier modes for the curves.
        numquadpoints (int): Number of quadrature points for the curves.

    Returns:
        list: list of CurveXYZFourier objects.
    """
    xyz = stel.XYZ0.T.detach().numpy() # (nphi, 3)

    # determine coil centers and axis tangent
    if not stellsym:
        end = stel.nphi + 1
    else:
        end = (stel.nphi + 1) // 2
    padding = end / ncurves / 2
    phi_idx = np.linspace(padding, end - padding, ncurves, dtype=int)
    centers = xyz[phi_idx] # (ncurves, 3)
    normals = stel.normal_cartesian.detach().numpy()[phi_idx] # (ncurves, 3)
    binormals = stel.binormal_cartesian.detach().numpy()[phi_idx] # (ncurves, 3)

    curves = []
    for i_curve in range(ncurves):
        curve = CurveXYZFourier(numquadpoints, order)
        # center the curve
        curve.set('xc(0)', centers[i_curve, 0])
        curve.set('yc(0)', centers[i_curve, 1])
        curve.set('zc(0)', centers[i_curve, 2])
        # orient the curve
        curve.set('xc(1)', R1 * normals[i_curve, 0])
        curve.set('yc(1)', R1 * normals[i_curve, 1])
        curve.set('zc(1)', R1 * normals[i_curve, 2])
        curve.set('xs(1)', R1 * binormals[i_curve, 0])
        curve.set('ys(1)', R1 * binormals[i_curve, 1])
        curve.set('zs(1)', R1 * binormals[i_curve, 2])
        curves.append(curve)

    return curves


def to_CurveRZFourier(stel):
    """Convert the magnetic axis of a Qsc object to a CurveRZFourier object.

    Args:
        stel (Qsc): Qsc object

    Returns:
        CurveRZFourier: magnetic axis as a CurveRZFourier object
    """
    from simsopt.geo import CurveRZFourier
    quadpoints = stel.phi.detach().numpy() / (2 * np.pi)
    nfourier = stel.nfourier
    nfp = stel.nfp
    stellsym = stel.stellsym
    curve = CurveRZFourier(quadpoints=quadpoints, order=nfourier - 1, nfp=nfp, stellsym=stellsym)
    curve.unfix_all()
    curve.x = np.concatenate((stel.rc.detach().numpy(), stel.zs[1:].detach().numpy())).flatten()
    return curve