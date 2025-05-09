"""
This module contains the calculation for the O(r^2) solution
"""

import logging
import numpy as np
from .util import mu0
import torch

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_r2(self):
    """
    Compute the O(r^2) quantities.
    """
    logger.debug('Calculating O(r^2) terms')
    # First, some shorthand:
    nphi = self.nphi
    B0 = self.B0
    G0 = torch.clone(self.G0)
    B0_over_abs_G0 = B0 / torch.abs(G0)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    X1c = torch.clone(self.X1c)
    Y1s = torch.clone(self.Y1s)
    Y1c = torch.clone(self.Y1c)
    sigma = torch.clone(self.sigma)
    d_d_varphi = torch.clone(self.d_d_varphi)
    iota_N = torch.clone(self.iotaN)
    iota = torch.clone(self.iota)
    curvature = torch.clone(self.curvature)
    torsion = torch.clone(self.torsion)
    etabar = torch.clone(self.etabar)
    I2 = torch.clone(self.I2)
    B2s = torch.clone(self.B2s)
    B2c = torch.clone(self.B2c)
    p2 = torch.clone(self.p2)
    sG = self.sG
    spsi = self.spsi
    I2_over_B0 = I2 / B0

    if torch.abs(iota_N) < 1e-8:
        logger.warning('|iota_N| is very small so O(r^2) solve will be poorly conditioned. '
                       f'iota_N={iota_N}')

    V1 = X1c * X1c + Y1c * Y1c + Y1s * Y1s
    V2 = 2 * Y1s * Y1c
    V3 = X1c * X1c + Y1c * Y1c - Y1s * Y1s

    factor = - B0_over_abs_G0 / 8
    Z20 = factor*torch.matmul(d_d_varphi,V1)
    Z2s = factor*(torch.matmul(d_d_varphi,V2) - 2 * iota_N * V3)
    Z2c = factor*(torch.matmul(d_d_varphi,V3) + 2 * iota_N * V2)

    qs = -iota_N * X1c - Y1s * torsion * abs_G0_over_B0
    qc = torch.matmul(d_d_varphi,X1c) - Y1c * torsion * abs_G0_over_B0
    rs = torch.matmul(d_d_varphi,Y1s) - iota_N * Y1c
    rc = torch.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * torsion * abs_G0_over_B0

    X2s = B0_over_abs_G0 * (torch.matmul(d_d_varphi,Z2s) - 2*iota_N*Z2c + B0_over_abs_G0 * ( abs_G0_over_B0*abs_G0_over_B0*B2s/B0 + (qc * qs + rc * rs)/2)) / curvature

    X2c = B0_over_abs_G0 * (torch.matmul(d_d_varphi,Z2c) + 2*iota_N*Z2s - B0_over_abs_G0 * (-abs_G0_over_B0*abs_G0_over_B0*B2c/B0 \
           + abs_G0_over_B0*abs_G0_over_B0*etabar*etabar/2 - (qc * qc - qs * qs + rc * rc - rs * rs)/4)) / curvature

    beta_1s = -4 * spsi * sG * mu0 * p2 * etabar * abs_G0_over_B0 / (iota_N * B0 * B0)

    Y2s_from_X20 = -sG * spsi * curvature * curvature / (etabar * etabar)
    Y2s_inhomogeneous = sG * spsi * (-curvature/2 + curvature*curvature/(etabar*etabar)*(-X2c + X2s * sigma))

    Y2c_from_X20 = -sG * spsi * curvature * curvature * sigma / (etabar * etabar)
    Y2c_inhomogeneous = sG * spsi * curvature * curvature / (etabar * etabar) * (X2s + X2c * sigma)

    # Note: in the fX* and fY* quantities below, I've omitted the
    # contributions from X20 and Y20 to the d/dzeta terms. These
    # contributions are handled later when we assemble the large
    # matrix.

    fX0_from_X20 = -4 * sG * spsi * abs_G0_over_B0 * (Y2c_from_X20 * Z2s - Y2s_from_X20 * Z2c)
    fX0_from_Y20 = -torsion * abs_G0_over_B0 - 4 * sG * spsi * abs_G0_over_B0 * (Z2s) \
        - spsi * I2_over_B0 * (-2) * abs_G0_over_B0
    fX0_inhomogeneous = curvature * abs_G0_over_B0 * Z20 - 4 * sG * spsi * abs_G0_over_B0 * (Y2c_inhomogeneous * Z2s - Y2s_inhomogeneous * Z2c) \
        - spsi * I2_over_B0 * (0.5 * curvature * sG * spsi) * abs_G0_over_B0 + beta_1s * abs_G0_over_B0 / 2 * Y1c

    fXs_from_X20 = -torsion * abs_G0_over_B0 * Y2s_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (Y2c_from_X20 * Z20) \
        - spsi * I2_over_B0 * (- 2 * Y2s_from_X20) * abs_G0_over_B0
    fXs_from_Y20 = - 4 * spsi * sG * abs_G0_over_B0 * (-Z2c + Z20)
    fXs_inhomogeneous = torch.matmul(d_d_varphi,X2s) - 2 * iota_N * X2c - torsion * abs_G0_over_B0 * Y2s_inhomogeneous + curvature * abs_G0_over_B0 * Z2s \
        - 4 * spsi * sG * abs_G0_over_B0 * (Y2c_inhomogeneous * Z20) \
        - spsi * I2_over_B0 * (0.5 * curvature * spsi * sG - 2 * Y2s_inhomogeneous) * abs_G0_over_B0 \
        - (0.5) * abs_G0_over_B0 * beta_1s * Y1s

    fXc_from_X20 = - torsion * abs_G0_over_B0 * Y2c_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (-Y2s_from_X20 * Z20) \
        - spsi * I2_over_B0 * (- 2 * Y2c_from_X20) * abs_G0_over_B0
    fXc_from_Y20 = - torsion * abs_G0_over_B0 - 4 * spsi * sG * abs_G0_over_B0 * (Z2s) \
        - spsi * I2_over_B0 * (-2) * abs_G0_over_B0
    fXc_inhomogeneous = torch.matmul(d_d_varphi,X2c) + 2 * iota_N * X2s - torsion * abs_G0_over_B0 * Y2c_inhomogeneous + curvature * abs_G0_over_B0 * Z2c \
        - 4 * spsi * sG * abs_G0_over_B0 * (-Y2s_inhomogeneous * Z20) \
        - spsi * I2_over_B0 * (0.5 * curvature * sG * spsi - 2 * Y2c_inhomogeneous) * abs_G0_over_B0 \
        - (0.5) * abs_G0_over_B0 * beta_1s * Y1c

    fY0_from_X20 = torsion * abs_G0_over_B0 - spsi * I2_over_B0 * (2) * abs_G0_over_B0
    fY0_from_Y20 = torch.zeros(nphi)
    fY0_inhomogeneous = -4 * spsi * sG * abs_G0_over_B0 * (X2s * Z2c - X2c * Z2s) \
        - spsi * I2_over_B0 * (-0.5 * curvature * X1c * X1c) * abs_G0_over_B0 - (0.5) * abs_G0_over_B0 * beta_1s * X1c

    fYs_from_X20 = -2 * iota_N * Y2c_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (Z2c)
    fYs_from_Y20 = torch.ones(nphi)* (-2 * iota_N)
    fYs_inhomogeneous = torch.matmul(d_d_varphi,Y2s_inhomogeneous) - 2 * iota_N * Y2c_inhomogeneous + torsion * abs_G0_over_B0 * X2s \
        - 4 * spsi * sG * abs_G0_over_B0 * (-X2c * Z20) - 2 * spsi * I2_over_B0 * X2s * abs_G0_over_B0

    fYc_from_X20 = 2 * iota_N * Y2s_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (-Z2s)
    fYc_from_Y20 = torch.zeros(nphi)
    fYc_inhomogeneous = torch.matmul(d_d_varphi,Y2c_inhomogeneous) + 2 * iota_N * Y2s_inhomogeneous + torsion * abs_G0_over_B0 * X2c \
        - 4 * spsi * sG * abs_G0_over_B0 * (X2s * Z20) \
        - spsi * I2_over_B0 * (-0.5 * curvature * X1c * X1c + 2 * X2c) * abs_G0_over_B0 + 0.5 * abs_G0_over_B0 * beta_1s * X1c

    matrix = torch.zeros((2 * nphi, 2 * nphi))
    right_hand_side = torch.zeros(2 * nphi)
    for j in range(nphi):
        # Handle the terms involving d X_0 / d zeta and d Y_0 / d zeta:
        # ----------------------------------------------------------------

        # Equation 1, terms involving X0:
        # Contributions arise from Y1c * fYs - Y1s * fYc.
        matrix[j, 0:nphi] = Y1c[j] * d_d_varphi[j, :] * Y2s_from_X20 - Y1s[j] * d_d_varphi[j, :] * Y2c_from_X20

        # Equation 1, terms involving Y0:
        # Contributions arise from -Y1s * fY0 - Y1s * fYc, and they happen to be equal.
        matrix[j, nphi:(2*nphi)] = -2 * Y1s[j] * d_d_varphi[j, :]

        # Equation 2, terms involving X0:
        # Contributions arise from -X1c * fX0 + Y1s * fYs + Y1c * fYc
        matrix[j+nphi, 0:nphi] = -X1c[j] * d_d_varphi[j, :] + Y1s[j] * d_d_varphi[j, :] * Y2s_from_X20 + Y1c[j] * d_d_varphi[j, :] * Y2c_from_X20

        # Equation 2, terms involving Y0:
        # Contributions arise from -Y1c * fY0 + Y1c * fYc, but they happen to cancel.

        # Now handle the terms involving X_0 and Y_0 without d/dzeta derivatives:
        # ----------------------------------------------------------------

        matrix[j, j       ] = matrix[j, j       ] + X1c[j] * fXs_from_X20[j] - Y1s[j] * fY0_from_X20[j] + Y1c[j] * fYs_from_X20[j] - Y1s[j] * fYc_from_X20[j]
        matrix[j, j + nphi] = matrix[j, j + nphi] + X1c[j] * fXs_from_Y20[j] - Y1s[j] * fY0_from_Y20[j] + Y1c[j] * fYs_from_Y20[j] - Y1s[j] * fYc_from_Y20[j]

        matrix[j + nphi, j       ] = matrix[j + nphi, j       ] - X1c[j] * fX0_from_X20[j] + X1c[j] * fXc_from_X20[j] - Y1c[j] * fY0_from_X20[j] + Y1s[j] * fYs_from_X20[j] + Y1c[j] * fYc_from_X20[j]
        matrix[j + nphi, j + nphi] = matrix[j + nphi, j + nphi] - X1c[j] * fX0_from_Y20[j] + X1c[j] * fXc_from_Y20[j] - Y1c[j] * fY0_from_Y20[j] + Y1s[j] * fYs_from_Y20[j] + Y1c[j] * fYc_from_Y20[j]


    right_hand_side[0:nphi] = -(X1c * fXs_inhomogeneous - Y1s * fY0_inhomogeneous + Y1c * fYs_inhomogeneous - Y1s * fYc_inhomogeneous)
    right_hand_side[nphi:2 * nphi] = -(- X1c * fX0_inhomogeneous + X1c * fXc_inhomogeneous - Y1c * fY0_inhomogeneous + Y1s * fYs_inhomogeneous + Y1c * fYc_inhomogeneous)

    solution = torch.linalg.solve(matrix, right_hand_side)
    X20 = solution[0:nphi]
    Y20 = solution[nphi:2 * nphi]

    # Now that we have X20 and Y20 explicitly, we can reconstruct Y2s, Y2c, and B20:
    Y2s = Y2s_inhomogeneous + Y2s_from_X20 * X20
    Y2c = Y2c_inhomogeneous + Y2c_from_X20 * X20 + Y20

    B20 = B0 * (curvature * X20 - B0_over_abs_G0 * torch.matmul(d_d_varphi,Z20) + (0.5) * etabar * etabar - mu0 * p2 / (B0 * B0) \
                - 0.25 * B0_over_abs_G0 * B0_over_abs_G0 * (qc * qc + qs * qs + rc * rc + rs * rs))

    d_l_d_phi = self.d_l_d_phi
    normalizer = 1 / torch.sum(d_l_d_phi)
    self.B20_mean = torch.sum(B20 * d_l_d_phi) * normalizer
    self.B20_anomaly = B20 - self.B20_mean
    self.B20_residual = torch.sqrt(torch.sum((B20 - self.B20_mean) * (B20 - self.B20_mean) * d_l_d_phi) * normalizer) / B0
    self.B20_variation = torch.max(B20) - torch.min(B20)

    self.N_helicity = - self.helicity * self.nfp
    self.G2 = -mu0 * p2 * G0 / (B0 * B0) - iota * I2

    self.d_curvature_d_varphi = torch.matmul(d_d_varphi, curvature)
    self.d_torsion_d_varphi = torch.matmul(d_d_varphi, torsion)
    self.d_X20_d_varphi = torch.matmul(d_d_varphi, X20)
    self.d_X2s_d_varphi = torch.matmul(d_d_varphi, X2s)
    self.d_X2c_d_varphi = torch.matmul(d_d_varphi, X2c)
    self.d_Y20_d_varphi = torch.matmul(d_d_varphi, Y20)
    self.d_Y2s_d_varphi = torch.matmul(d_d_varphi, Y2s)
    self.d_Y2c_d_varphi = torch.matmul(d_d_varphi, Y2c)
    self.d_Z20_d_varphi = torch.matmul(d_d_varphi, Z20)
    self.d_Z2s_d_varphi = torch.matmul(d_d_varphi, Z2s)
    self.d_Z2c_d_varphi = torch.matmul(d_d_varphi, Z2c)
    self.d2_X1c_d_varphi2 = torch.matmul(d_d_varphi, self.d_X1c_d_varphi)
    self.d2_Y1c_d_varphi2 = torch.matmul(d_d_varphi, self.d_Y1c_d_varphi)
    self.d2_Y1s_d_varphi2 = torch.matmul(d_d_varphi, self.d_Y1s_d_varphi)

    # Store all important results in self:
    self.V1 = V1
    self.V2 = V2
    self.V3 = V3

    self.X20 = X20
    self.X2s = X2s
    self.X2c = X2c
    self.Y20 = Y20
    self.Y2s = Y2s
    self.Y2c = Y2c
    self.Z20 = Z20
    self.Z2s = Z2s
    self.Z2c = Z2c
    self.beta_1s = beta_1s
    self.B20 = B20

    # O(r^2) diagnostics:
    self.mercier()
    self.calculate_grad_grad_B_tensor()
    
    # TODO: @padidar torch the following and uncomment
    # self.calculate_r_singularity()

    if self.helicity == 0:
        self.X20_untwisted = self.X20
        self.X2s_untwisted = self.X2s
        self.X2c_untwisted = self.X2c
        self.Y20_untwisted = self.Y20
        self.Y2s_untwisted = self.Y2s
        self.Y2c_untwisted = self.Y2c
        self.Z20_untwisted = self.Z20
        self.Z2s_untwisted = self.Z2s
        self.Z2c_untwisted = self.Z2c
    else:
        angle = -self.helicity * self.nfp * self.varphi
        sinangle = torch.sin(angle)
        cosangle = torch.cos(angle)
        self.X20_untwisted = self.X20
        self.Y20_untwisted = self.Y20
        self.Z20_untwisted = self.Z20
        sinangle = torch.sin(2*angle)
        cosangle = torch.cos(2*angle)
        self.X2s_untwisted = self.X2s *   cosangle  + self.X2c * sinangle
        self.X2c_untwisted = self.X2s * (-sinangle) + self.X2c * cosangle
        self.Y2s_untwisted = self.Y2s *   cosangle  + self.Y2c * sinangle
        self.Y2c_untwisted = self.Y2s * (-sinangle) + self.Y2c * cosangle
        self.Z2s_untwisted = self.Z2s *   cosangle  + self.Z2c * sinangle
        self.Z2c_untwisted = self.Z2s * (-sinangle) + self.Z2c * cosangle

def calculate_r2_vac(self):
    """
    Compute the vacuum component of the O(r^2) quantities.
    """
    logger.debug('Calculating vacuum O(r^2) terms')
    # First, some shorthand:
    nphi = self.nphi
    B0 = self.B0
    G0 = torch.clone(self.G0)
    B0_over_abs_G0 = B0 / torch.abs(G0)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    X1c = torch.clone(self.X1c)
    Y1s = torch.clone(self.Y1s)
    Y1c = torch.clone(self.Y1c_vac) # use vacuum value
    sigma = torch.clone(self.sigma_vac) # use vacuum value
    d_d_varphi = torch.clone(self.d_d_varphi)
    iota_N = torch.clone(self.iotaN_vac) # use vacuum value
    iota = torch.clone(self.iota_vac) # use vacuum value
    curvature = torch.clone(self.curvature)
    torsion = torch.clone(self.torsion)
    etabar = torch.clone(self.etabar)
    I2 = 0.0 # no current in vacuum
    B2s = torch.clone(self.B2s)
    B2c = torch.clone(self.B2c)
    p2 = 0.0 # no pressure gradient in vacuum
    sG = self.sG
    spsi = self.spsi
    I2_over_B0 = I2 / self.B0

    if torch.abs(iota_N) < 1e-8:
        logger.warning('|iota_N| is very small so O(r^2) solve will be poorly conditioned. '
                       f'iota_N={iota_N}')

    V1 = X1c * X1c + Y1c * Y1c + Y1s * Y1s
    V2 = 2 * Y1s * Y1c
    V3 = X1c * X1c + Y1c * Y1c - Y1s * Y1s

    factor = - B0_over_abs_G0 / 8
    Z20 = factor*torch.matmul(d_d_varphi,V1)
    Z2s = factor*(torch.matmul(d_d_varphi,V2) - 2 * iota_N * V3)
    Z2c = factor*(torch.matmul(d_d_varphi,V3) + 2 * iota_N * V2)

    qs = -iota_N * X1c - Y1s * torsion * abs_G0_over_B0
    qc = torch.matmul(d_d_varphi,X1c) - Y1c * torsion * abs_G0_over_B0
    rs = torch.matmul(d_d_varphi,Y1s) - iota_N * Y1c
    rc = torch.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * torsion * abs_G0_over_B0

    X2s = B0_over_abs_G0 * (torch.matmul(d_d_varphi,Z2s) - 2*iota_N*Z2c + B0_over_abs_G0 * ( abs_G0_over_B0*abs_G0_over_B0*B2s/B0 + (qc * qs + rc * rs)/2)) / curvature

    X2c = B0_over_abs_G0 * (torch.matmul(d_d_varphi,Z2c) + 2*iota_N*Z2s - B0_over_abs_G0 * (-abs_G0_over_B0*abs_G0_over_B0*B2c/B0 \
           + abs_G0_over_B0*abs_G0_over_B0*etabar*etabar/2 - (qc * qc - qs * qs + rc * rc - rs * rs)/4)) / curvature

    beta_1s = -4 * spsi * sG * mu0 * p2 * etabar * abs_G0_over_B0 / (iota_N * B0 * B0)

    Y2s_from_X20 = -sG * spsi * curvature * curvature / (etabar * etabar)
    Y2s_inhomogeneous = sG * spsi * (-curvature/2 + curvature*curvature/(etabar*etabar)*(-X2c + X2s * sigma))

    Y2c_from_X20 = -sG * spsi * curvature * curvature * sigma / (etabar * etabar)
    Y2c_inhomogeneous = sG * spsi * curvature * curvature / (etabar * etabar) * (X2s + X2c * sigma)

    # Note: in the fX* and fY* quantities below, I've omitted the
    # contributions from X20 and Y20 to the d/dzeta terms. These
    # contributions are handled later when we assemble the large
    # matrix.

    fX0_from_X20 = -4 * sG * spsi * abs_G0_over_B0 * (Y2c_from_X20 * Z2s - Y2s_from_X20 * Z2c)
    fX0_from_Y20 = -torsion * abs_G0_over_B0 - 4 * sG * spsi * abs_G0_over_B0 * (Z2s) \
        - spsi * I2_over_B0 * (-2) * abs_G0_over_B0
    fX0_inhomogeneous = curvature * abs_G0_over_B0 * Z20 - 4 * sG * spsi * abs_G0_over_B0 * (Y2c_inhomogeneous * Z2s - Y2s_inhomogeneous * Z2c) \
        - spsi * I2_over_B0 * (0.5 * curvature * sG * spsi) * abs_G0_over_B0 + beta_1s * abs_G0_over_B0 / 2 * Y1c

    fXs_from_X20 = -torsion * abs_G0_over_B0 * Y2s_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (Y2c_from_X20 * Z20) \
        - spsi * I2_over_B0 * (- 2 * Y2s_from_X20) * abs_G0_over_B0
    fXs_from_Y20 = - 4 * spsi * sG * abs_G0_over_B0 * (-Z2c + Z20)
    fXs_inhomogeneous = torch.matmul(d_d_varphi,X2s) - 2 * iota_N * X2c - torsion * abs_G0_over_B0 * Y2s_inhomogeneous + curvature * abs_G0_over_B0 * Z2s \
        - 4 * spsi * sG * abs_G0_over_B0 * (Y2c_inhomogeneous * Z20) \
        - spsi * I2_over_B0 * (0.5 * curvature * spsi * sG - 2 * Y2s_inhomogeneous) * abs_G0_over_B0 \
        - (0.5) * abs_G0_over_B0 * beta_1s * Y1s

    fXc_from_X20 = - torsion * abs_G0_over_B0 * Y2c_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (-Y2s_from_X20 * Z20) \
        - spsi * I2_over_B0 * (- 2 * Y2c_from_X20) * abs_G0_over_B0
    fXc_from_Y20 = - torsion * abs_G0_over_B0 - 4 * spsi * sG * abs_G0_over_B0 * (Z2s) \
        - spsi * I2_over_B0 * (-2) * abs_G0_over_B0
    fXc_inhomogeneous = torch.matmul(d_d_varphi,X2c) + 2 * iota_N * X2s - torsion * abs_G0_over_B0 * Y2c_inhomogeneous + curvature * abs_G0_over_B0 * Z2c \
        - 4 * spsi * sG * abs_G0_over_B0 * (-Y2s_inhomogeneous * Z20) \
        - spsi * I2_over_B0 * (0.5 * curvature * sG * spsi - 2 * Y2c_inhomogeneous) * abs_G0_over_B0 \
        - (0.5) * abs_G0_over_B0 * beta_1s * Y1c

    fY0_from_X20 = torsion * abs_G0_over_B0 - spsi * I2_over_B0 * (2) * abs_G0_over_B0
    fY0_from_Y20 = torch.zeros(nphi)
    fY0_inhomogeneous = -4 * spsi * sG * abs_G0_over_B0 * (X2s * Z2c - X2c * Z2s) \
        - spsi * I2_over_B0 * (-0.5 * curvature * X1c * X1c) * abs_G0_over_B0 - (0.5) * abs_G0_over_B0 * beta_1s * X1c

    fYs_from_X20 = -2 * iota_N * Y2c_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (Z2c)
    fYs_from_Y20 = torch.ones(nphi)* (-2 * iota_N)
    fYs_inhomogeneous = torch.matmul(d_d_varphi,Y2s_inhomogeneous) - 2 * iota_N * Y2c_inhomogeneous + torsion * abs_G0_over_B0 * X2s \
        - 4 * spsi * sG * abs_G0_over_B0 * (-X2c * Z20) - 2 * spsi * I2_over_B0 * X2s * abs_G0_over_B0

    fYc_from_X20 = 2 * iota_N * Y2s_from_X20 - 4 * spsi * sG * abs_G0_over_B0 * (-Z2s)
    fYc_from_Y20 = torch.zeros(nphi)
    fYc_inhomogeneous = torch.matmul(d_d_varphi,Y2c_inhomogeneous) + 2 * iota_N * Y2s_inhomogeneous + torsion * abs_G0_over_B0 * X2c \
        - 4 * spsi * sG * abs_G0_over_B0 * (X2s * Z20) \
        - spsi * I2_over_B0 * (-0.5 * curvature * X1c * X1c + 2 * X2c) * abs_G0_over_B0 + 0.5 * abs_G0_over_B0 * beta_1s * X1c

    matrix = torch.zeros((2 * nphi, 2 * nphi))
    right_hand_side = torch.zeros(2 * nphi)
    for j in range(nphi):
        # Handle the terms involving d X_0 / d zeta and d Y_0 / d zeta:
        # ----------------------------------------------------------------

        # Equation 1, terms involving X0:
        # Contributions arise from Y1c * fYs - Y1s * fYc.
        matrix[j, 0:nphi] = Y1c[j] * d_d_varphi[j, :] * Y2s_from_X20 - Y1s[j] * d_d_varphi[j, :] * Y2c_from_X20

        # Equation 1, terms involving Y0:
        # Contributions arise from -Y1s * fY0 - Y1s * fYc, and they happen to be equal.
        matrix[j, nphi:(2*nphi)] = -2 * Y1s[j] * d_d_varphi[j, :]

        # Equation 2, terms involving X0:
        # Contributions arise from -X1c * fX0 + Y1s * fYs + Y1c * fYc
        matrix[j+nphi, 0:nphi] = -X1c[j] * d_d_varphi[j, :] + Y1s[j] * d_d_varphi[j, :] * Y2s_from_X20 + Y1c[j] * d_d_varphi[j, :] * Y2c_from_X20

        # Equation 2, terms involving Y0:
        # Contributions arise from -Y1c * fY0 + Y1c * fYc, but they happen to cancel.

        # Now handle the terms involving X_0 and Y_0 without d/dzeta derivatives:
        # ----------------------------------------------------------------

        matrix[j, j       ] = matrix[j, j       ] + X1c[j] * fXs_from_X20[j] - Y1s[j] * fY0_from_X20[j] + Y1c[j] * fYs_from_X20[j] - Y1s[j] * fYc_from_X20[j]
        matrix[j, j + nphi] = matrix[j, j + nphi] + X1c[j] * fXs_from_Y20[j] - Y1s[j] * fY0_from_Y20[j] + Y1c[j] * fYs_from_Y20[j] - Y1s[j] * fYc_from_Y20[j]

        matrix[j + nphi, j       ] = matrix[j + nphi, j       ] - X1c[j] * fX0_from_X20[j] + X1c[j] * fXc_from_X20[j] - Y1c[j] * fY0_from_X20[j] + Y1s[j] * fYs_from_X20[j] + Y1c[j] * fYc_from_X20[j]
        matrix[j + nphi, j + nphi] = matrix[j + nphi, j + nphi] - X1c[j] * fX0_from_Y20[j] + X1c[j] * fXc_from_Y20[j] - Y1c[j] * fY0_from_Y20[j] + Y1s[j] * fYs_from_Y20[j] + Y1c[j] * fYc_from_Y20[j]


    right_hand_side[0:nphi] = -(X1c * fXs_inhomogeneous - Y1s * fY0_inhomogeneous + Y1c * fYs_inhomogeneous - Y1s * fYc_inhomogeneous)
    right_hand_side[nphi:2 * nphi] = -(- X1c * fX0_inhomogeneous + X1c * fXc_inhomogeneous - Y1c * fY0_inhomogeneous + Y1s * fYs_inhomogeneous + Y1c * fYc_inhomogeneous)

    solution = torch.linalg.solve(matrix, right_hand_side)
    X20 = solution[0:nphi]
    Y20 = solution[nphi:2 * nphi]

    # Now that we have X20 and Y20 explicitly, we can reconstruct Y2s, Y2c, and B20:
    Y2s = Y2s_inhomogeneous + Y2s_from_X20 * X20
    Y2c = Y2c_inhomogeneous + Y2c_from_X20 * X20 + Y20

    B20 = B0 * (curvature * X20 - B0_over_abs_G0 * torch.matmul(d_d_varphi,Z20) + (0.5) * etabar * etabar - mu0 * p2 / (B0 * B0) \
                - 0.25 * B0_over_abs_G0 * B0_over_abs_G0 * (qc * qc + qs * qs + rc * rc + rs * rs))

    d_l_d_phi = self.d_l_d_phi
    normalizer = 1 / torch.sum(d_l_d_phi)
    # self.B20_mean = torch.sum(B20 * d_l_d_phi) * normalizer
    # self.B20_anomaly = B20 - self.B20_mean
    # self.B20_residual = torch.sqrt(torch.sum((B20 - self.B20_mean) * (B20 - self.B20_mean) * d_l_d_phi) * normalizer) / B0
    # self.B20_variation = torch.max(B20) - torch.min(B20)

    # self.N_helicity = - self.helicity * self.nfp
    self.G2_vac = -mu0 * p2 * G0 / (B0 * B0) - iota * I2

    # self.d_curvature_d_varphi = torch.matmul(d_d_varphi, curvature)
    # self.d_torsion_d_varphi = torch.matmul(d_d_varphi, torsion)
    self.d_X20_vac_d_varphi = torch.matmul(d_d_varphi, X20)
    self.d_X2s_vac_d_varphi = torch.matmul(d_d_varphi, X2s)
    self.d_X2c_vac_d_varphi = torch.matmul(d_d_varphi, X2c)
    self.d_Y20_vac_d_varphi = torch.matmul(d_d_varphi, Y20)
    self.d_Y2s_vac_d_varphi = torch.matmul(d_d_varphi, Y2s)
    self.d_Y2c_vac_d_varphi = torch.matmul(d_d_varphi, Y2c)
    self.d_Z20_vac_d_varphi = torch.matmul(d_d_varphi, Z20)
    self.d_Z2s_vac_d_varphi = torch.matmul(d_d_varphi, Z2s)
    self.d_Z2c_vac_d_varphi = torch.matmul(d_d_varphi, Z2c)
    # self.d2_X1c_d_varphi2 = torch.matmul(d_d_varphi, self.d_X1c_d_varphi)
    self.d2_Y1c_vac_d_varphi2 = torch.matmul(d_d_varphi, self.d_Y1c_vac_d_varphi)
    # self.d2_Y1s_d_varphi2 = torch.matmul(d_d_varphi, self.d_Y1s_d_varphi)

    # Store all important results in self:
    self.V1_vac = V1
    self.V2_vac = V2
    self.V3_vac = V3

    self.X20_vac = X20
    self.X2s_vac = X2s
    self.X2c_vac = X2c
    self.Y20_vac = Y20
    self.Y2s_vac = Y2s
    self.Y2c_vac = Y2c
    self.Z20_vac = Z20
    self.Z2s_vac = Z2s
    self.Z2c_vac = Z2c
    self.beta_1s_vac = beta_1s
    self.B20_vac = B20

    # O(r^2) diagnostics:
    self.calculate_grad_grad_B_tensor_vac()

    if self.helicity == 0:
        self.X20_vac_untwisted = self.X20_vac
        self.X2s_vac_untwisted = self.X2s_vac
        self.X2c_vac_untwisted = self.X2c_vac
        self.Y20_vac_untwisted = self.Y20_vac
        self.Y2s_vac_untwisted = self.Y2s_vac
        self.Y2c_vac_untwisted = self.Y2c_vac
        self.Z20_vac_untwisted = self.Z20_vac
        self.Z2s_vac_untwisted = self.Z2s_vac
        self.Z2c_vac_untwisted = self.Z2c_vac
    else:
        angle = -self.helicity * self.nfp * self.varphi
        sinangle = torch.sin(angle)
        cosangle = torch.cos(angle)
        self.X20_vac_untwisted = self.X20_vac
        self.Y20_vac_untwisted = self.Y20_vac
        self.Z20_vac_untwisted = self.Z20_vac
        sinangle = torch.sin(2*angle)
        cosangle = torch.cos(2*angle)
        self.X2s_vac_untwisted = self.X2s_vac *   cosangle  + self.X2c_vac * sinangle
        self.X2c_vac_untwisted = self.X2s_vac * (-sinangle) + self.X2c_vac * cosangle
        self.Y2s_vac_untwisted = self.Y2s_vac *   cosangle  + self.Y2c_vac * sinangle
        self.Y2c_vac_untwisted = self.Y2s_vac * (-sinangle) + self.Y2c_vac * cosangle
        self.Z2s_vac_untwisted = self.Z2s_vac *   cosangle  + self.Z2c_vac * sinangle
        self.Z2c_vac_untwisted = self.Z2s_vac * (-sinangle) + self.Z2c_vac * cosangle
