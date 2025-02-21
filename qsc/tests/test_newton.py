#!/usr/bin/env python3

import unittest
import numpy as np
import torch
from qsc.newton import newton

class NewtonTests(unittest.TestCase):

    def test_2d(self):
        """
        Try a little 2D nonlinear example.
        """
        torch.set_default_dtype(torch.float64)
        def func(x):
            return torch.tensor([x[1] - torch.exp(x[0]), x[0] + x[1]])
        
        def jac(x):
            return torch.tensor([[-torch.exp(x[0]), 1],
                             [1, 1]])
        x0 = torch.zeros(2)
        soln = newton(func, x0, jac)
        np.testing.assert_allclose(soln, [-0.5671432904097838, 0.5671432904097838])
        
if __name__ == "__main__":
    unittest.main()
