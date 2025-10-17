import numpy as np
"""
This module contains methods for constructing a Qsc object 
from a VMEC equilibrium.
"""

@classmethod
def from_vmec(cls, v, n_fourier=5, **kwargs):
    """Build a Qsc object from a VMEC equilibria. Since Qsc represents quasisymmetric
    equilbria, the VMEC equilibria should be close to quasisymmetric on the magnetic axis.
    Furthermore, if pressure and current profiles are used that have large O(s^2) terms,
    the resulting Qsc object may not accurately represent the equilibrium.

    Currently only stellarator-symmetric equilibria are supported.

    Example:
        stel = Qsc.from_vmec(v)

    Args
    ----
        v (Vmec): An instance of the Simsopt Vmec class.
        n_fourier (int): Highest Fourier mode number to use for the axis shape.
        **kwargs: Additional keyword arguments passed to the Qsc constructor.

    Returns
    -------
        self (Qsc): The Qsc object.
    """
    from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry
    from simsopt.geo import CurveRZFourier
    from simsopt.mhd import Boozer

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

    print('nfp', nfp)
    print('stellsym', stellsym)

    s = np.linspace(1e-8, 1, 32, endpoint=True)
    ntheta_vmec = 63
    nphi_vmec = 2 * n_fourier + 1
    theta = np.linspace(0, 2 * np.pi, ntheta_vmec)
    phi = np.linspace(0, 2 * np.pi / nfp, nphi_vmec)
    data = vmec_compute_geometry(v, s, theta, phi)

    # extract the axis shape
    X = data.X[0].reshape((ntheta_vmec, nphi_vmec)).mean(axis=0) # (nphi,)
    Y = data.Y[0].reshape((ntheta_vmec, nphi_vmec)).mean(axis=0) # (nphi,)
    Z = data.Z[0].reshape((ntheta_vmec, nphi_vmec)).mean(axis=0) # (nphi,)

    # VMEC uses a different sign convention for Z
    Z = -Z

    xyz = np.array([X, Y, Z]).T # (nphi, 3)
    curve = CurveRZFourier(quadpoints=phi/2/np.pi, order = n_fourier, nfp=nfp, stellsym=stellsym)
    curve.unfix_all()
    curve.least_squares_fit(xyz)
    rc = curve.x[:n_fourier+1]
    zs = np.append([0.0], curve.x[n_fourier+1:])

    print('rc')
    print(rc)
    print('zs')
    print(zs)

    ## sanity check: plot the axis shape
    # from simsopt.geo import plot
    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # curve1 = CurveRZFourier(quadpoints=99, order = n_fourier, nfp=nfp, stellsym=stellsym)
    # curve1.x = curve.x
    # add_default_args(kwargs, rc=rc, zs=zs, nfp=nfp, etabar=0.1, B0=1,
    #              nphi=99, order="r1")
    # stel =  cls(**kwargs)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # xyz = stel.XYZ0.detach().numpy()
    # ax.plot(xyz[0], xyz[1], xyz[2], label='Axis Shape')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plot([curve1], ax = ax, show=True)

    # Get B0
    print('modB shape', data.modB.shape)
    modB = data.modB[0] # (ntheta, nphi)
    B0 = modB.mean()
    print("B0", B0)

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
        curtor = 2 * np.pi / (mu0 * I2 * a**2).
    """
    phiedge = v.wout.phi[-1]
    # # TODO: check sign of toroidal flux vs sign of phiedge
    # print("toroidal_flux_sign", data.toroidal_flux_sign)
    # print("sign(phiedge)", np.sign(phiedge))

    # get the minor radius through phiedge = spsi * pi * a^2 * B0
    # spsi = data.toroidal_flux_sign
    spsi = np.sign(phiedge)
    aminor = np.sqrt(np.abs(phiedge) / (np.pi * B0))

    dp_ds = data.d_pressure_d_s # (ns,)
    p2 = (dp_ds[0] * spsi * np.pi * B0) / phiedge
    print('dp_ds', dp_ds)

    # get I2
    # TODO: double check ctor is curtor
    print('curtor', v.wout.ctor)
    mu0 = 4 * np.pi * 1e-7
    I2 = (v.wout.ctor * mu0 * aminor**2) / (2 * np.pi)
    if np.isclose(I2, 0.0, atol=1e-15):
        I2 = 0.0
    print('I2', I2)

    booz = Boozer(v)
    def modB_boozer(rs, ntheta=65, nphi=65):
        
        s = (rs / aminor) ** 2
        booz.register(s) # register surface
        booz.run()
        bx = booz.bx

        # discretize boozer angles
        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        phi, theta = np.meshgrid(phi1d, theta1d, indexing='ij')

        # reconstruct |B|
        modB = np.zeros((len(s), *np.shape(phi)))
        for js in range(len(s)):
            for jmn in range(len(bx.xm_b)):
                m = bx.xm_b[jmn]
                n = bx.xn_b[jmn]
                angle = m * theta - n * phi
                modB[js] += bx.bmnc_b[jmn, js] * np.cos(angle)
                if bx.asym:
                    modB[js] += bx.bmns_b[jmn, js] * np.sin(angle)

        return modB # (nphi, ntheta)
    
    # TODO: what surface should we do this on?
    r = aminor
    ntheta_booz = 65
    nphi_booz = 65
    modB = modB_boozer(np.array([r]), ntheta=ntheta_booz, nphi=nphi_booz)[0] # (ntheta, nphi)

    """
    etabar is a free parameter for the first order expansion.
    We get it from the BoozXform since
        B = B0 * (1 + etabar * r * cos(theta) + O(r^2))
    """
    # get etabar
    B_theta_fft = np.fft.fft((modB / B0 - 1) / r, axis=-1)
    etabar = (2 / ntheta_booz) * np.real(B_theta_fft[:, 1]).mean()

    """
    Next we will get B2c and B2s which are constants of the field strength
    B(r, theta, phi) expansion to second order in r:
        B(r, theta, phi) = B0 * (1 + etabar * r * cos(theta) + r^2 * (B2c * cos(2*theta) + B2s * sin(2*theta)) + O(r^3)
    
    For stellarator-symmetric equilibria, B2s = 0.
    """
    # compute B2c and B2s
    B1 = B0 * (1 + etabar * r * np.cos(np.linspace(0, 2 * np.pi, ntheta_booz, endpoint=False)))
    B_theta_fft = np.fft.fft((modB - B1) / r / r, axis=-1)
    B2c = (2 / ntheta_booz) * np.real(B_theta_fft[:, 2]).mean() 
    if stellsym:
        B2s = 0.0
    else:
        B2s = (-2 / ntheta_booz) * np.imag(B_theta_fft[:, 2]).mean()
    print('B2c', B2c)
    print('B2s', B2s)

    # TODO: do we need to set sG?

    add_default_args(kwargs, rc=rc, zs=zs, nfp=nfp, etabar=etabar, B0=B0,
                 I2=I2, nphi=99, B2s=B2s, B2c=B2c, p2=p2, order="r3",
                 spsi = spsi)
    stel = cls(**kwargs)

    print('vmec iota', data.iota[0])
    print('stel iota', stel.iota)

    return stel