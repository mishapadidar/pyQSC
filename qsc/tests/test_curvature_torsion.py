#!/usr/bin/env python3

import unittest
import os
from scipy.io import netcdf
import numpy as np
import logging
from qsc.qsc import Qsc
from qsc.util import to_Fourier

def test_curvature_torsion():
    """
    Test that the curvature and torsion match an independent
    calculation using the fortran code.
    """
    
    # Stellarator-symmetric case:
    stel = Qsc(rc=[1.3, 0.3, 0.01, -0.001],
                zs=[0, 0.4, -0.02, -0.003], nfp=5, nphi=15)
    
    curvature_fortran = [1.74354628565018, 1.61776632275718, 1.5167042487094, 
                            1.9179603622369, 2.95373444883134, 3.01448808361584, 1.7714523990583, 
                            1.02055493647363, 1.02055493647363, 1.77145239905828, 3.01448808361582, 
                            2.95373444883135, 1.91796036223691, 1.5167042487094, 1.61776632275717]
    
    torsion_fortran = [0.257226801231061, -0.131225053326418, -1.12989287766591, 
                        -1.72727988032403, -1.48973327005739, -1.34398161921833, 
                        -1.76040161697108, -2.96573007082039, -2.96573007082041, 
                        -1.7604016169711, -1.34398161921833, -1.48973327005739, 
                        -1.72727988032403, -1.12989287766593, -0.13122505332643]

    varphi_fortran = [0, 0.0909479184372571, 0.181828299105257, 
                        0.268782689120682, 0.347551637441381, 0.42101745128188, 
                        0.498195826255542, 0.583626271820683, 0.673010789615233, 
                        0.758441235180374, 0.835619610154036, 0.909085423994535, 
                        0.987854372315234, 1.07480876233066, 1.16568914299866]

    rtol = 1e-13
    atol = 1e-13
    print(np.allclose(stel.curvature.detach().numpy(), curvature_fortran, rtol=rtol, atol=atol))
    print(np.allclose(stel.torsion.detach().numpy(), torsion_fortran, rtol=rtol, atol=atol))
    print(np.allclose(stel.varphi.detach().numpy(), varphi_fortran, rtol=rtol, atol=atol))

if __name__ == "__main__":
    test_curvature_torsion()