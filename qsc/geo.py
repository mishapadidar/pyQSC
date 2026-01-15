#!/usr/bin/env python3

"""
Methods for computing the flux surface geometry
"""
from functools import lru_cache
import logging
import numpy as np
import torch
from .util import rotate_nfp, Struct

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=2)
def _load_components(self, vacuum_component=False):
    """Build a Struct object with the untwisted Fourier components of the flux
    surface shape, X1c_untwisted, X1s_untwisted, etc as attributes. If vacuum_component
    the vacuum components X1c_untwisted_vac, X1s_untwisted_vac, etc are set instead, though
    they are still accessed without the _vac suffix.

    Example:
        components = self._load_components()
        X20_untwisted = components.X20_untwisted
        components_vac = self._load_components(vacuum_component = True)
        X2c_untwisted = components.X2c_untwisted

    Args:
        vacuum_component (bool): if True, the vacuum components X1c_untwisted, X1s_untwisted, etc
        are set at attributes instead. Default False.

    Returns:
        Struct: a struct with attributes X1c_untwisted, Y1c_untwisted, etc.
    """
    
    variables = ['X1c',
                 'Y1c',
                 'X1s',
                 'Y1s',
                 ]

    if self.order != 'r1':
        variables += ['X20',
                      'Y20',
                      'Z20',
                      'X2c',
                      'Y2c',
                      'Z2c',
                      'X2s',
                      'Y2s',
                      'Z2s'
                      ]
        if self.order == 'r3':
                variables += ['X3c1',
                              'X3s1',
                              'X3c3',
                              'X3s3',
                              'Y3c1',
                              'Y3s1',
                              'Y3c3',
                              'Y3s3',
                              'Z3c1',
                              'Z3s1',
                              'Z3c3',
                              'Z3s3'
                              ]

    if vacuum_component:
        suffix = '_vac_untwisted'
    else:
        suffix = '_untwisted'

    # TODO: does Struct need to be torch?
    out = Struct
    for v in variables:
        # TODO: do we need to torch.clone?
        s = 'self.'+ v + suffix
        setattr(out, v + '_untwisted', eval(s))

    return out

@lru_cache(maxsize=8)
def surface(self, r, ntheta=64, vacuum_component=False):
    """Compute points on a flux surface with radius r. The quadrature points are 
    uniformly spaced in Boozer poloidal angle, theta, and the axis cylindrical angle,
    phi0.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of poloidal quadrature points. Defaults to 64.
            The number of phi quadpoints is inherited from the class's nphi
            attribute.
        vacuum_component (bool): if True, the surface shape of the vacuum equilirium (p2=I2=0)
            is computed. Default False.

    Returns:
        tensor: (nphi, ntheta, 3) tensor of points (x,y,z) on a flux surface.
    """

    # axis
    xyz0 = self.XYZ0 # (3, nphi)

    # frenet-frame
    t = self.tangent_cartesian.T # (3, nphi)
    n = self.normal_cartesian.T
    b = self.binormal_cartesian.T

    # theta = torch.tensor(np.linspace(0, 2 * np.pi, ntheta, endpoint=False))
    theta =torch.linspace(0, 2 * torch.pi, ntheta+1)[:-1]

    # storage
    xyz = torch.zeros((self.nphi, ntheta, 3))

    components = self._load_components(vacuum_component=vacuum_component)

    for j_theta in range(ntheta):
        costheta = torch.cos(theta[j_theta])
        sintheta = torch.sin(theta[j_theta])
        X_at_this_theta = r * (components.X1c_untwisted * costheta + components.X1s_untwisted * sintheta)
        Y_at_this_theta = r * (components.Y1c_untwisted * costheta + components.Y1s_untwisted * sintheta)
        Z_at_this_theta = 0 * X_at_this_theta

        if self.order != 'r1':
            # We need O(r^2) terms:
            cos2theta = torch.cos(2 * theta[j_theta])
            sin2theta = torch.sin(2 * theta[j_theta])
            X_at_this_theta += r * r * (components.X20_untwisted + components.X2c_untwisted * cos2theta + components.X2s_untwisted * sin2theta)
            Y_at_this_theta += r * r * (components.Y20_untwisted + components.Y2c_untwisted * cos2theta + components.Y2s_untwisted * sin2theta)
            Z_at_this_theta += r * r * (components.Z20_untwisted + components.Z2c_untwisted * cos2theta + components.Z2s_untwisted * sin2theta)

            if self.order == 'r3':
                # We need O(r^3) terms:
                costheta  = torch.cos(theta[j_theta])
                sintheta  = torch.sin(theta[j_theta])
                cos3theta = torch.cos(3 * theta[j_theta])
                sin3theta = torch.sin(3 * theta[j_theta])
                r3 = r * r * r
                X_at_this_theta += r3 * (components.X3c1_untwisted * costheta + components.X3s1_untwisted * sintheta
                                        + components.X3c3_untwisted * cos3theta + components.X3s3_untwisted * sin3theta)
                Y_at_this_theta += r3 * (components.Y3c1_untwisted * costheta + components.Y3s1_untwisted * sintheta
                                        + components.Y3c3_untwisted * cos3theta + components.Y3s3_untwisted * sin3theta)
                Z_at_this_theta += r3 * (components.Z3c1_untwisted * costheta + components.Z3s1_untwisted * sintheta
                                        + components.Z3c3_untwisted * cos3theta + components.Z3s3_untwisted * sin3theta)

        point = xyz0 + X_at_this_theta * n + Y_at_this_theta * b + Z_at_this_theta * t # (3, nphi)

        xyz[:,j_theta,:] = point.T # (nphi, ntheta, 3)
            
    return xyz

@lru_cache(maxsize=8)
def dsurface_by_dvarphi(self, r, ntheta=64, vacuum_component=False):
    """Compute the derivative of the flux surface map surface(r, theta, varphi) with respect to varphi.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadpoints. Defaults to 64.
        vacuum_component (bool): if True, the derivative of the vacuum surface (p2=I2=0)
            is computed. Default False.

    Returns:
        tensor: (nphi, ntheta, 3) tensor of vectors dsurface/dvarphi on a flux surface.
    """

    # derivative of axis wrt varphi
    dxyz0_by_dvarphi = self.dXYZ0_by_dphi / self.d_varphi_d_phi # (3, nphi)

    # axis
    xyz0 = self.XYZ0 # (3, nphi)

    # frenet-frame
    t = self.tangent_cartesian.T # (3, nphi)
    n = self.normal_cartesian.T
    b = self.binormal_cartesian.T

    components = self._load_components(vacuum_component=vacuum_component)

    """
    Use the chain rule to get the derivatives of the frenet-frame wrt varphi, i.e.
        dt/dvarphi = dt/ds * ds/dvarphi = kappa * n * ds/dvarphi
        dn/dvarphi = dn/ds * ds/dvarphi = (-kappa * t + tau * b) * ds/dvarphi
        db/dvarphi = db/ds * ds/dvarphi = (-tau * n) * ds/dvarphi
    where s is the arc length.
    """
    dt_by_dvarphi = self.curvature * n * self.d_l_d_varphi # (3, nphi)
    dn_by_dvarphi = (- self.curvature * t + self.torsion * b) * self.d_l_d_varphi # (3, nphi)
    db_by_dvarphi = (- self.torsion * n) * self.d_l_d_varphi # (3, nphi)

    # spectral diff the coefficients
    # order r1
    dX1c_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.X1c_untwisted)
    dX1s_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.X1s_untwisted)
    dY1c_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.Y1c_untwisted)
    dY1s_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.Y1s_untwisted)

    if self.order != 'r1':
        # We need O(r^2) terms:
        dX20_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.X20_untwisted)
        dX2c_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.X2c_untwisted)
        dX2s_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.X2s_untwisted)
        dY20_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.Y20_untwisted)
        dY2c_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.Y2c_untwisted)
        dY2s_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.Y2s_untwisted)
        dZ20_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.Z20_untwisted)
        dZ2c_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.Z2c_untwisted)
        dZ2s_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.Z2s_untwisted)

        if self.order == 'r3':
            # We need O(r^3) terms:
            dX3c1_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.X3c1_untwisted)
            dX3s1_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.X3s1_untwisted)
            dX3c3_untwisted_by_dvarphi = 0.0 # X3c3 is hardcoded to zero in PyQSC
            dX3s3_untwisted_by_dvarphi = 0.0 # X3s3 is hardcoded to zero in PyQSC
            dY3c1_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.Y3c1_untwisted)
            dY3s1_untwisted_by_dvarphi = torch.matmul(self.d_d_varphi, components.Y3s1_untwisted)
            dY3c3_untwisted_by_dvarphi = 0.0 # Y3c3 is hardcoded to zero in PyQSC
            dY3s3_untwisted_by_dvarphi = 0.0 # Y3s3 is hardcoded to zero in PyQSC
            dZ3c1_untwisted_by_dvarphi = 0.0 # Z3c1 is hardcoded to zero in PyQSC
            dZ3s1_untwisted_by_dvarphi = 0.0 # Z3s1 is hardcoded to zero in PyQSC
            dZ3c3_untwisted_by_dvarphi = 0.0 # Z3c3 is hardcoded to zero in PyQSC
            dZ3s3_untwisted_by_dvarphi = 0.0 # Z3s3 is hardcoded to zero in PyQSC

    # theta = torch.tensor(np.linspace(0, 2 * torch.pi, ntheta, endpoint=False))
    theta =torch.linspace(0, 2 * torch.pi, ntheta+1)[:-1]

    # storage
    xyz = torch.zeros((self.nphi, ntheta, 3))

    for j_theta in range(ntheta):
        costheta = torch.cos(theta[j_theta])
        sintheta = torch.sin(theta[j_theta])
        X_at_this_theta = r * (components.X1c_untwisted * costheta + components.X1s_untwisted * sintheta)
        Y_at_this_theta = r * (components.Y1c_untwisted * costheta + components.Y1s_untwisted * sintheta)
        Z_at_this_theta = 0 * X_at_this_theta

        dX_by_dvarphi_at_this_theta = r * (dX1c_untwisted_by_dvarphi * costheta + dX1s_untwisted_by_dvarphi * sintheta)
        dY_by_dvarphi_at_this_theta = r * (dY1c_untwisted_by_dvarphi * costheta + dY1s_untwisted_by_dvarphi * sintheta)
        dZ_by_dvarphi_at_this_theta = 0 * dX_by_dvarphi_at_this_theta

        if self.order != 'r1':
            # We need O(r^2) terms:
            cos2theta = torch.cos(2 * theta[j_theta])
            sin2theta = torch.sin(2 * theta[j_theta])
            X_at_this_theta += r * r * (components.X20_untwisted + components.X2c_untwisted * cos2theta + components.X2s_untwisted * sin2theta)
            Y_at_this_theta += r * r * (components.Y20_untwisted + components.Y2c_untwisted * cos2theta + components.Y2s_untwisted * sin2theta)
            Z_at_this_theta += r * r * (components.Z20_untwisted + components.Z2c_untwisted * cos2theta + components.Z2s_untwisted * sin2theta)

            dX_by_dvarphi_at_this_theta += r * r * (dX20_untwisted_by_dvarphi + dX2c_untwisted_by_dvarphi * cos2theta + dX2s_untwisted_by_dvarphi * sin2theta)
            dY_by_dvarphi_at_this_theta += r * r * (dY20_untwisted_by_dvarphi + dY2c_untwisted_by_dvarphi * cos2theta + dY2s_untwisted_by_dvarphi * sin2theta)
            dZ_by_dvarphi_at_this_theta += r * r * (dZ20_untwisted_by_dvarphi + dZ2c_untwisted_by_dvarphi * cos2theta + dZ2s_untwisted_by_dvarphi * sin2theta)

            if self.order == 'r3':
                # We need O(r^3) terms:
                costheta  = torch.cos(theta[j_theta])
                sintheta  = torch.sin(theta[j_theta])
                cos3theta = torch.cos(3 * theta[j_theta])
                sin3theta = torch.sin(3 * theta[j_theta])
                r3 = r * r * r
                X_at_this_theta += r3 * (components.X3c1_untwisted * costheta + components.X3s1_untwisted * sintheta
                                        + components.X3c3_untwisted * cos3theta + components.X3s3_untwisted * sin3theta)
                Y_at_this_theta += r3 * (components.Y3c1_untwisted * costheta + components.Y3s1_untwisted * sintheta
                                        + components.Y3c3_untwisted * cos3theta + components.Y3s3_untwisted * sin3theta)
                Z_at_this_theta += r3 * (components.Z3c1_untwisted * costheta + components.Z3s1_untwisted * sintheta
                                        + components.Z3c3_untwisted * cos3theta + components.Z3s3_untwisted * sin3theta)

                dX_by_dvarphi_at_this_theta += r3 * (dX3c1_untwisted_by_dvarphi * costheta + dX3s1_untwisted_by_dvarphi * sintheta
                                        + dX3c3_untwisted_by_dvarphi * cos3theta + dX3s3_untwisted_by_dvarphi * sin3theta)
                dY_by_dvarphi_at_this_theta += r3 * (dY3c1_untwisted_by_dvarphi * costheta + dY3s1_untwisted_by_dvarphi * sintheta
                                        + dY3c3_untwisted_by_dvarphi * cos3theta + dY3s3_untwisted_by_dvarphi * sin3theta)
                dZ_by_dvarphi_at_this_theta += r3 * (dZ3c1_untwisted_by_dvarphi * costheta + dZ3s1_untwisted_by_dvarphi * sintheta
                                        + dZ3c3_untwisted_by_dvarphi * cos3theta + dZ3s3_untwisted_by_dvarphi * sin3theta)

        point = (dxyz0_by_dvarphi + X_at_this_theta * dn_by_dvarphi + Y_at_this_theta * db_by_dvarphi + Z_at_this_theta * dt_by_dvarphi +
                    dX_by_dvarphi_at_this_theta * n + dY_by_dvarphi_at_this_theta * b + dZ_by_dvarphi_at_this_theta * t) # (3, nphi)

        xyz[:,j_theta,:] = point.T # (nphi, ntheta, 3)

    return xyz

@lru_cache(maxsize=8)
def dsurface_by_dtheta(self, r, ntheta=64, vacuum_component=False):
    """Compute the derivative of the flux surface map with respect to the
    Boozer poloidal angle, theta.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadpoints. Defaults to 64.
        vacuum_component (bool): if True, the derivative of the vacuum surface (p2=I2=0)
            is computed. Default False.

    Returns:
        tensor: (nphi, ntheta, 3) tensor of vectors dsurface/dtheta on a flux surface.
    """

    # frenet-frame
    t = self.tangent_cartesian.T # (3, nphi)
    n = self.normal_cartesian.T
    b = self.binormal_cartesian.T

    # theta = torch.tensor(np.linspace(0, 2 * np.pi, ntheta, endpoint=False))
    theta =torch.linspace(0, 2 * torch.pi, ntheta+1)[:-1]

    components = self._load_components(vacuum_component=vacuum_component)

    # storage
    xyz = torch.zeros((self.nphi, ntheta, 3))

    for j_theta in range(ntheta):
        costheta = -torch.sin(theta[j_theta])
        sintheta = torch.cos(theta[j_theta])
        X_at_this_theta = r * (components.X1c_untwisted * costheta + components.X1s_untwisted * sintheta)
        Y_at_this_theta = r * (components.Y1c_untwisted * costheta + components.Y1s_untwisted * sintheta)
        Z_at_this_theta = 0 * X_at_this_theta

        if self.order != 'r1':
            # We need O(r^2) terms:
            cos2theta = -2 * torch.sin(2 * theta[j_theta])
            sin2theta = 2 * torch.cos(2 * theta[j_theta])
            X_at_this_theta += r * r * (components.X2c_untwisted * cos2theta + components.X2s_untwisted * sin2theta)
            Y_at_this_theta += r * r * (components.Y2c_untwisted * cos2theta + components.Y2s_untwisted * sin2theta)
            Z_at_this_theta += r * r * (components.Z2c_untwisted * cos2theta + components.Z2s_untwisted * sin2theta)

            if self.order == 'r3':
                # We need O(r^3) terms:
                cos3theta = - 3 * torch.sin(3 * theta[j_theta])
                sin3theta = 3 * torch.cos(3 * theta[j_theta])
                r3 = r * r * r
                X_at_this_theta += r3 * (components.X3c1_untwisted * costheta + components.X3s1_untwisted * sintheta
                                        + components.X3c3_untwisted * cos3theta + components.X3s3_untwisted * sin3theta)
                Y_at_this_theta += r3 * (components.Y3c1_untwisted * costheta + components.Y3s1_untwisted * sintheta
                                        + components.Y3c3_untwisted * cos3theta + components.Y3s3_untwisted * sin3theta)
                Z_at_this_theta += r3 * (components.Z3c1_untwisted * costheta + components.Z3s1_untwisted * sintheta
                                        + components.Z3c3_untwisted * cos3theta + components.Z3s3_untwisted * sin3theta)

        point = X_at_this_theta * n + Y_at_this_theta * b + Z_at_this_theta * t # (3, nphi)

        xyz[:,j_theta,:] = point.T
            
    return xyz

@lru_cache(maxsize=8)
def d2surface_by_dthetatheta(self, r, ntheta=64, vacuum_component=False):
    """Compute the second derivative of the flux surface map with respect to the
    Boozer poloidal angle, theta.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadpoints. Defaults to 64.
        vacuum_component (bool): if True, the derivative of the vacuum surface (p2=I2=0)
            is computed. Default False.

    Returns:
        tensor: (nphi, ntheta, 3) tensor of vectors dsurface/dtheta on a flux surface.
    """
    # frenet-frame
    t = self.tangent_cartesian.T # (3, nphi)
    n = self.normal_cartesian.T
    b = self.binormal_cartesian.T

    # theta = torch.tensor(np.linspace(0, 2 * np.pi, ntheta, endpoint=False))
    theta =torch.linspace(0, 2 * torch.pi, ntheta+1)[:-1]


    components = self._load_components(vacuum_component=vacuum_component)

    # storage
    xyz = torch.zeros((self.nphi, ntheta, 3))

    for j_theta in range(ntheta):
        costheta = - torch.cos(theta[j_theta])
        sintheta = - torch.sin(theta[j_theta])
        X_at_this_theta = r * (components.X1c_untwisted * costheta + components.X1s_untwisted * sintheta)
        Y_at_this_theta = r * (components.Y1c_untwisted * costheta + components.Y1s_untwisted * sintheta)
        Z_at_this_theta = 0 * X_at_this_theta

        if self.order != 'r1':
            # We need O(r^2) terms:
            cos2theta = -4 * torch.cos(2 * theta[j_theta])
            sin2theta = -4 * torch.sin(2 * theta[j_theta])
            X_at_this_theta += r * r * (components.X2c_untwisted * cos2theta + components.X2s_untwisted * sin2theta)
            Y_at_this_theta += r * r * (components.Y2c_untwisted * cos2theta + components.Y2s_untwisted * sin2theta)
            Z_at_this_theta += r * r * (components.Z2c_untwisted * cos2theta + components.Z2s_untwisted * sin2theta)

            if self.order == 'r3':
                # We need O(r^3) terms:
                cos3theta = -9 * torch.cos(3 * theta[j_theta])
                sin3theta = -9 * torch.sin(3 * theta[j_theta])
                r3 = r * r * r
                X_at_this_theta += r3 * (components.X3c1_untwisted * costheta + components.X3s1_untwisted * sintheta
                                        + components.X3c3_untwisted * cos3theta + components.X3s3_untwisted * sin3theta)
                Y_at_this_theta += r3 * (components.Y3c1_untwisted * costheta + components.Y3s1_untwisted * sintheta
                                        + components.Y3c3_untwisted * cos3theta + components.Y3s3_untwisted * sin3theta)
                Z_at_this_theta += r3 * (components.Z3c1_untwisted * costheta + components.Z3s1_untwisted * sintheta
                                        + components.Z3c3_untwisted * cos3theta + components.Z3s3_untwisted * sin3theta)

        point = X_at_this_theta * n + Y_at_this_theta * b + Z_at_this_theta * t # (3, nphi)

        xyz[:,j_theta,:] = point.T
            
    return xyz

@lru_cache(maxsize=8)
def dsurface_by_dr(self, r, ntheta=64, vacuum_component=False):
    """Compute the derivative of the flux surface map with respect to the
    minor radius coordinate, r.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadpoints. Defaults to 64.
        vacuum_component (bool): if True, the derivative of the vacuum surface (p2=I2=0)
            is computed. Default False.

    Returns:
        tensor: (nphi, ntheta, 3) tensor of vectors dsurface/dr on a flux surface.
    """

    # axis
    xyz0 = self.XYZ0 # (3, nphi)

    # frenet-frame
    t = self.tangent_cartesian.T # (3, nphi)
    n = self.normal_cartesian.T
    b = self.binormal_cartesian.T

    # theta = torch.tensor(np.linspace(0, 2 * np.pi, ntheta, endpoint=False))
    theta =torch.linspace(0, 2 * torch.pi, ntheta+1)[:-1]

    components = self._load_components(vacuum_component=vacuum_component)

    # storage
    xyz = torch.zeros((self.nphi, ntheta, 3))

    for j_theta in range(ntheta):
        costheta = torch.cos(theta[j_theta])
        sintheta = torch.sin(theta[j_theta])
        X_at_this_theta = (components.X1c_untwisted * costheta + components.X1s_untwisted * sintheta)
        Y_at_this_theta = (components.Y1c_untwisted * costheta + components.Y1s_untwisted * sintheta)
        Z_at_this_theta = 0 * X_at_this_theta

        if self.order != 'r1':
            # We need O(r^2) terms:
            cos2theta = torch.cos(2 * theta[j_theta])
            sin2theta = torch.sin(2 * theta[j_theta])
            X_at_this_theta += 2 * r * (components.X20_untwisted + components.X2c_untwisted * cos2theta + components.X2s_untwisted * sin2theta)
            Y_at_this_theta += 2 * r * (components.Y20_untwisted + components.Y2c_untwisted * cos2theta + components.Y2s_untwisted * sin2theta)
            Z_at_this_theta += 2 * r * (components.Z20_untwisted + components.Z2c_untwisted * cos2theta + components.Z2s_untwisted * sin2theta)

            if self.order == 'r3':
                # We need O(r^3) terms:
                costheta  = torch.cos(theta[j_theta])
                sintheta  = torch.sin(theta[j_theta])
                cos3theta= torch.cos(3 * theta[j_theta])
                sin3theta= torch.sin(3 * theta[j_theta])
                r3 = 3 * r * r
                X_at_this_theta += r3 * (components.X3c1_untwisted * costheta + components.X3s1_untwisted * sintheta
                                        + components.X3c3_untwisted * cos3theta + components.X3s3_untwisted * sin3theta)
                Y_at_this_theta += r3 * (components.Y3c1_untwisted * costheta + components.Y3s1_untwisted * sintheta
                                        + components.Y3c3_untwisted * cos3theta + components.Y3s3_untwisted * sin3theta)
                Z_at_this_theta += r3 * (components.Z3c1_untwisted * costheta + components.Z3s1_untwisted * sintheta
                                        + components.Z3c3_untwisted * cos3theta + components.Z3s3_untwisted * sin3theta)

        point = X_at_this_theta * n + Y_at_this_theta * b + Z_at_this_theta * t # (3, nphi)

        xyz[:,j_theta,:] = point.T # (nphi, ntheta, 3)
            
    return xyz

@lru_cache(maxsize=8)
def surface_normal(self, r, ntheta=64, vacuum_component=False):
    """Compute the normal vectors to a flux surface with radius r.
    
    We respect the variable ordering (r, theta, varphi) for cross products
    and covariant/contravariant representation.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 64.
            The number of phi quadpoints is inherited from the class's nphi
            attribute.
        vacuum_component (bool): if True, the normal vectors of the vacuum surface (p2=I2=0)
            are computed. Default False.

    Returns:
        tensor: (nphi, ntheta, 3) tensor of normal vectors.
    """
    
    gd1 = self.dsurface_by_dvarphi(r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
    gd2 = self.dsurface_by_dtheta(r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
    normal = torch.linalg.cross(gd2, gd1, dim=-1) # (nphi, ntheta, 3)
    return normal

@lru_cache(maxsize=8)
def jacobian(self, r, ntheta=64, vacuum_component=False):
    """Compute the Jacobian of the the coordinate transformation from (r, theta, varphi),
        sqrt{g} = d(surface)/dr * (d(surface)/dtheta x d(surface)/dvarphi).
        
    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 64.
            The number of phi quadpoints is inherited from the class's nphi
            attribute.
        vacuum_component (bool): if True, the Jacobian of the vacuum surface (p2=I2=0)
            is computed. Default False.

    Returns:
        tensor: (nphi, ntheta) tensor of Jacobian determinants.
    """
    
    normal = self.surface_normal(r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
    dsurface_by_dr = self.dsurface_by_dr(r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
    jacobian_det = torch.sum(dsurface_by_dr * normal, axis=-1) # (nphi, ntheta)
    return jacobian_det

@lru_cache(maxsize=8)
def surface_area_element(self, r, ntheta=64, vacuum_component=False):
    """Compute the area element with respect to theta and varphi (not phi) of a flux surface with radius r.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 64.
            The number of phi quadpoints is inherited from the class's nphi
            attribute.
        vacuum_component (bool): if True, the area element of the vacuum surface (p2=I2=0)
            is computed. Default False.

    Returns:
        tensor: (nphi, ntheta) tensor containing the area element dA = ||d(surface)/dtheta x d(surface)/dvarphi||.
    """
    normal = self.surface_normal(r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
    dA = torch.linalg.norm(normal, dim=-1) # (nphi, ntheta)
    return torch.clone(dA)

@lru_cache(maxsize=8)
def surface_area(self, r, ntheta=64, vacuum_component=False):
    """Compute the area of a flux surface with radius r.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 64.
            The number of phi quadpoints is inherited from the class's nphi
            attribute.
        vacuum_component (bool): if True, the area of the vacuum surface (p2=I2=0)
            is computed. Default False.

    Returns:
        tensor: (1,) tensor containing the area of the flux surface.
    """
    X = torch.ones(self.nphi, ntheta) # (nphi, ntheta)
    area = self.surface_integral(X, r, vacuum_component=vacuum_component)
    return area

@lru_cache(maxsize=8)
def surface_theta_curvature(self, r, ntheta=64, vacuum_component=False):
    """Compute the curvature of a flux surface, with radius r, in the theta direction,
        kappa = || dsurface/dtheta x d^2surface/(dtheta^2)|| / ||dsurface/dtheta||^3.

    Args:
        r (float): radius of flux surface
        ntheta (int, optional): number of theta quadrature points. Defaults to 64.
            The number of phi quadpoints is inherited from the class's nphi
            attribute.
        vacuum_component (bool): if True, the curvature of the vacuum surface (p2=I2=0)
            is computed. Default False.

    Returns:
        tensor: (nphi, ntheta) tensor of curvature values.
    """
    
    d1 = self.dsurface_by_dtheta(r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
    d2 = self.d2surface_by_dthetatheta(r, ntheta=ntheta, vacuum_component=vacuum_component) # (nphi, ntheta, 3)
    cross = torch.linalg.cross(d1, d2, dim=-1) # (nphi, ntheta, 3)
    numerator = torch.linalg.norm(cross, dim=-1) # (nphi, ntheta)
    denominator = torch.linalg.norm(d1, dim=-1)**3 # (nphi, ntheta)
    curvature = numerator / denominator # (nphi, ntheta)
    return curvature