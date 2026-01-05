import numpy as np
"""
This module contains methods for constructing a Qsc object 
from a VMEC equilibrium.
"""

@classmethod
def from_vmec(cls, v, n_fourier=10, **kwargs):
    """Build a Qsc object from a VMEC equilibria. To ensure a good fit, the
    VMEC equilibria should be precisely quasisymmetric on the magnetic axis.
    Furthermore, if pressure and current profiles are used that have large O(s^2) terms,
    the resulting Qsc object may not accurately represent the equilibrium.

    NOTE: The Boozer poloidal angle used by BoozXform differs from the
    near-axis poloidal angle by a shift and sign reversal. This is accounted for
    in the calculation of etabar, B2c, and B2s below. This means that iota_vmec
    and iota_nae differ by a sign, and furthermore that the contours of |B|
    will be mirrored in the poloidal angle.

    Currently only stellarator-symmetric equilibria are supported.

    Example:
        stel = Qsc.from_vmec(v)

    Args
    ----
        v (Vmec): An instance of the Simsopt Vmec class.
        n_fourier (int): Highest Fourier mode number to use for the axis shape.
            Default is 10.
        **kwargs: Additional keyword arguments passed to the Qsc constructor.

    Returns
    -------
        self (Qsc): The Qsc object.
    """
    try:
        from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry
        from simsopt.geo import CurveRZFourier
        from simsopt.mhd import Boozer
    except ImportError:
        raise ImportError("The 'simsopt' library is required to use this from_vmec.")

    # ensure VMEC has run
    v.run()

    def add_default_args(kwargs_old, **kwargs_new):
        """
        Take any key-value arguments in ``kwargs_new`` and treat them as
        defaults, adding them to the dict ``kwargs_old`` only if
        they are not specified there.
        """
        for key in kwargs_new:
            if key not in kwargs_old:
                kwargs_old[key] = kwargs_new[key]


    nfp = v.wout.nfp
    stellsym = not v.wout.lasym

    if not stellsym:
        raise NotImplementedError("from_vmec currently only supports stellarator-symmetric configurations.")

    s = np.linspace(1e-8, 1, 64, endpoint=True)
    ntheta_vmec = 63
    nphi_vmec = 2 * n_fourier + 1
    theta_vmec = np.linspace(0, 2 * np.pi, ntheta_vmec)
    phi_vmec = np.linspace(0, 2 * np.pi / nfp, nphi_vmec)
    data = vmec_compute_geometry(v, s, theta_vmec, phi_vmec)

    # extract the axis shape
    X = data.X[0].reshape((ntheta_vmec, nphi_vmec)).mean(axis=0) # (nphi,)
    Y = data.Y[0].reshape((ntheta_vmec, nphi_vmec)).mean(axis=0) # (nphi,)
    Z = data.Z[0].reshape((ntheta_vmec, nphi_vmec)).mean(axis=0) # (nphi,)

    # TODO: make sure the axis is increasing in phi. Otherwise
    # TODO: we need to reorder the points.

    xyz = np.array([X, Y, Z]).T # (nphi, 3)
    curve = CurveRZFourier(quadpoints=phi_vmec/2/np.pi, order = n_fourier, nfp=nfp, stellsym=stellsym)
    curve.unfix_all()
    curve.least_squares_fit(xyz)
    rc = curve.x[:n_fourier+1]
    zs = np.append([0.0], curve.x[n_fourier+1:])

    # Get B0
    modB = data.modB[0] # (ntheta, nphi)
    B0 = modB.mean()

    # get the profiles
    """
    The toroidal flux is
        |phiedge| = pi * a^2 * B0
    from this we get spsi and the minor radius, a.

    The vmec pressure profile is
        p(r) = p0 + p2 * r^2 + O(r^4)
    with s = spsi * pi * r^2 * B0 / phiedge, so that 
        p(s) = p0 + p2 * s * phiedge / (spsi * pi * B0) + O(r^4)
    Then,
        dp/ds(s=0) = p2 * phiedge / (spsi * pi * B0)
    so,
        p2 = dp/ds(s=0) * (spsi * pi * B0) / phiedge

    The near axis current profile is 
        I(r) = I2 * r^2.
    We get the I2 from curtor,
        curtor = (2 * np.pi / mu0) * I2 * a**2.
    """

    # get the minor radius through phiedge = spsi * pi * a^2 * B0
    phiedge = v.wout.phi[-1]
    spsi = np.sign(phiedge)
    aminor = np.sqrt(np.abs(phiedge) / (np.pi * B0))

    dp_ds = data.d_pressure_d_s # (ns,)
    p2 = (dp_ds[0] * spsi * np.pi * B0) / phiedge

    # get I2
    mu0 = 4 * np.pi * 1e-7
    I2 = (v.wout.ctor * mu0) / (2 * np.pi * aminor**2)
    if np.isclose(I2, 0.0, atol=1e-15):
        I2 = 0.0

    """
    Now we compute the field strength in boozer coordinates using the BoozXform class.
    We will use this to determine etabar, B2c, and B2s.
    """

    booz = Boozer(v)
    def modB_boozer(r, ntheta=65, nphi=65):
        """Compute |B| in Boozer coordinates.

        The BoozXform theta differs from the near-axis theta by a shift and
        reversal, 
            theta_booz = - theta_nae + pi.
        We determined this empirically from the cross sections and |B| from the 
        BoozXform and the NAE.
        

        Args:
            rs (float): minor radius to compute |B| on.
            ntheta (int, optional): number of theta quadpoints. Defaults to 65.
            nphi (int, optional): number of phi quadpoints. Defaults to 65.

        Returns:
            (B, R, Z): (nphi, ntheta) arrays of |B|, R, and Z on the specified surface.
        """
        
        s = (r / aminor) ** 2
        booz.register([s]) # register surface
        booz.run()
        bx = booz.bx

        # discretize boozer angles
        theta1d = np.linspace(3*np.pi, np.pi, ntheta, endpoint=False)
        # theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        phi, theta = np.meshgrid(phi1d, theta1d, indexing='ij')

        # reconstruct |B|
        modB = np.zeros(np.shape(phi))
        R = np.zeros(np.shape(phi))
        Z = np.zeros(np.shape(phi))
        js = 0
        for jmn in range(len(bx.xm_b)):
            m = bx.xm_b[jmn]
            n = bx.xn_b[jmn]
            angle = m * theta - n * phi
            modB += bx.bmnc_b[jmn, js] * np.cos(angle)
            R += bx.rmnc_b[jmn, js] * np.cos(angle)
            Z += bx.zmns_b[jmn, js] * np.sin(angle)
            if bx.asym:
                modB += bx.bmns_b[jmn, js] * np.sin(angle)
                R += bx.rmns_b[jmn, js] * np.sin(angle)
                Z += bx.zmnc_b[jmn, js] * np.cos(angle)

        return modB, R, Z # (nphi, ntheta)
    
    # analysis is on LCFS
    r = aminor
    ntheta_booz = 65
    nphi_booz = 63
    modB, _, _ = modB_boozer(r, ntheta=ntheta_booz, nphi=nphi_booz) # (nphi, ntheta)

    # determine sG
    sG = np.sign(booz.bx.Boozer_G[0])

    # determine helicity and N
    temp = cls(rc=rc, zs=zs, nfp=nfp, B0=B0, order="r1", spsi=spsi, sG=sG)
    helicity = round(-temp.helicity.detach().numpy())
    N = helicity * nfp

    """
    etabar is a free parameter for the first order expansion.
    We determine it from the BoozXform since
        B = B0 * (1 + etabar * r * cos(vartheta) + O(r^2))
    where vartheta = theta - N * phi is the helical angle.
    We compute etabar by a least-squares fit.
    """
    # theta1d is the NAE theta
    theta1d = np.linspace(0, 2 * np.pi, ntheta_booz, endpoint=False)
    phi1d = np.linspace(0, 2 * np.pi / nfp, nphi_booz, endpoint=False)
    phi, theta = np.meshgrid(phi1d, theta1d, indexing='ij')
    vartheta = theta - N * phi
    x = B0 * r * np.cos(vartheta)
    etabar = np.mean((modB - B0) * x) / np.mean(x**2)

    """
    Next we will get B2c and B2s which are constants of the field strength
    B(r, theta, phi) expansion to second order in r:
        B(r, theta, phi) = B0 * (1 + etabar * r * cos(vartheta) 
            + r^2 * (B20(varphi) + B2c * cos(2 * vartheta) + B2s * sin(2 * vartheta)) + O(r^3)
    
    For stellarator-symmetric equilibria, B2s = 0.
    We compute B2c and B2s by a least-squares fit.
    """
    modB_shifted = modB - (B0 * (1 + r * etabar * np.cos(vartheta)) )
    B20_times_r_squared = np.mean(modB_shifted, axis=1, keepdims=True) # (nphi, 1)
    modB_shifted = modB_shifted - B20_times_r_squared
    x = r**2 * np.cos(2 * (vartheta))
    B2c = np.mean((modB_shifted) * x) / np.mean(x**2)
    if stellsym:
        B2s = 0.0
    else:
        x = r * r * np.sin(2 * (vartheta))
        B2s = np.mean((modB_shifted) * x) / np.mean(x**2)

    # rule of thumb for nphi
    nphi = 9 * n_fourier + 1

    add_default_args(kwargs, rc=rc, zs=zs, nfp=nfp, etabar=etabar, B0=B0,
                 I2=I2, nphi=nphi, B2s=B2s, B2c=B2c, p2=p2, order="r3",
                 spsi=spsi, sG=sG)
    stel = cls(**kwargs)

    return stel