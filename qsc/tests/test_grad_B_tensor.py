#!/usr/bin/env python3

import unittest
import numpy as np
import logging
from qsc.qsc import Qsc
import torch
from qsc.util import finite_difference_torch


logger = logging.getLogger(__name__)

class GradBTensorTests(unittest.TestCase):

    def test_axisymmetry(self):
        """
        Test that the grad B tensor is correct for an axisymmetric vacuum field
        """
        for sG in [-1, 1]:
            for spsi in [-1, 1]:
                for nfp in range(1, 4):
                    for j in range(3):
                        R0 = np.random.rand() * 2 + 0.3
                        B0 = np.random.rand() * 2 + 0.3
                        nphi = int(np.random.rand() * 10 + 3)
                        etabar = (np.random.rand() - 0.5) * 4
                        stel = Qsc(rc=[R0], zs=[0], nfp=nfp, nphi=nphi, sG=sG, \
                                   spsi=spsi, B0=B0, etabar=etabar)
                        val = sG * B0 / R0
                        if np.mod(nphi, 2) == 0:
                            nphi += 1

                        rtol = 1e-13
                        atol = 1e-13
                        np.testing.assert_allclose(stel.grad_B_tensor.nt, np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stel.grad_B_tensor.tn, np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stel.grad_B_tensor.nn, np.zeros(nphi), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stel.grad_B_tensor.bb, np.zeros(nphi), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stel.grad_B_tensor.nb, np.zeros(nphi), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stel.grad_B_tensor.bn, np.zeros(nphi), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(stel.L_grad_B, np.full(nphi, R0), rtol=rtol, atol=atol)
                        
class GradGradBTensorTests(unittest.TestCase):

    def test_alternate_derivation_and_symmetry(self):
        """
        Compare Rogerio's derivation to mine to ensure the results
        coincide.  Also verify symmetry in the first two components.
        """
        for sG in [-1, 1]:
            for spsi in [-1, 1]:
                for config in [1, 2, 3, 4, 5]:
                    B0 = np.random.rand() * 0.4 + 0.8
                    nphi = int(np.random.rand() * 50 + 61)
                    s = Qsc.from_paper(config, sG=sG, spsi=spsi, B0=B0, nphi=65)
                    s.calculate_grad_grad_B_tensor(two_ways=True)
                    logger.info("Max difference between Matt and Rogerio's derivation for config {} is {}".format(config, np.max(np.abs(s.grad_grad_B - s.grad_grad_B_alt))))
                    np.testing.assert_allclose(s.grad_grad_B, s.grad_grad_B_alt, rtol=1e-7, atol=1e-7)
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                # For all configs, the tensor should be symmetric in the 1st 2 indices:
                                np.testing.assert_allclose(s.grad_grad_B[:, i, j, k], s.grad_grad_B[:, j, i, k], rtol=1e-8, atol=1e-8)
                                # For curl-free fields, the tensor should also be symmetric in the last 2 indices:
                                if config in {1, 2, 4}:
                                    np.testing.assert_allclose(s.grad_grad_B[:, i, j, k], s.grad_grad_B[:, i, k, j], rtol=1e-8, atol=1e-8)
            
    def test_axisymmetry(self):
        """
        Test that the grad grad B tensor is correct for an axisymmetric vacuum field
        """
        B0_vals = [0.9, 1.1]
        R0_vals = [0.85, 1.0, 1.2]
        nphi_vals = [3, 5, 7, 9, 11]
        etabar_vals = [0.55, 0.65, 0.95, 1.0, 1.09, 1.3, 1.4]
        index = 0
        for sG in [-1, 1]:
            for spsi in [-1, 1]:
                for nfp in range(1, 4):
                    for j in range(3):
                        """
                        R0 = np.random.rand() * 2 + 0.3
                        B0 = np.random.rand() * 2 + 0.3
                        nphi = int(np.random.rand() * 10 + 3)
                        etabar = (np.random.rand() - 0.5) * 4
                        """
                        # Try many combinations of values, in a deterministic way:
                        R0 = R0_vals[np.mod(index, len(R0_vals))]
                        B0 = B0_vals[np.mod(index, len(B0_vals))]
                        nphi = nphi_vals[np.mod(index, len(nphi_vals))]
                        etabar = etabar_vals[np.mod(index, len(etabar_vals))]
                        index += 1
                        
                        s = Qsc(rc=[R0], zs=[0], nfp=nfp, nphi=nphi, sG=sG, \
                                   spsi=spsi, B0=B0, etabar=etabar, I2=1.e-8, order='r2')
                        # The O(r^2) solve fails if iota is exactly 0,
                        # so above we set a tiny nonzero current to
                        # create a tiny nonzero iota.
                        if np.mod(nphi, 2) == 0:
                            nphi += 1

                        s.calculate_grad_grad_B_tensor(two_ways=True)
                        # Check all components other than {tnn, ntn, nnt, ttt}. These should be 0.
                        rtol = 1e-6
                        atol = 1e-6
                        for i in range(3):
                            for j in range(3):
                                for k in range(3):
                                    if not ((i==2 and j==2 and k==2) or (i==2 and j==0 and k==0) or (i==0 and j==2 and k==0) or (i==i and j==0 and k==2)):
                                        np.testing.assert_allclose(s.grad_grad_B[:,i,j,k], np.zeros(nphi), rtol=rtol, atol=atol)
                                        np.testing.assert_allclose(s.grad_grad_B_alt[:,i,j,k], np.zeros(nphi), rtol=rtol, atol=atol)
                                        
                        # Check ttt component
                        val = -2 * sG * B0 / (R0 * R0)
                        np.testing.assert_allclose(s.grad_grad_B[:,2,2,2], np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(s.grad_grad_B_alt[:,2,2,2], np.full(nphi, val), rtol=rtol, atol=atol)

                        # Check tnn, ntn, and nnt components:
                        val = 2 * sG * B0 / (R0 * R0)
                        np.testing.assert_allclose(s.grad_grad_B[:,2,0,0], np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(s.grad_grad_B[:,0,2,0], np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(s.grad_grad_B[:,0,0,2], np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(s.grad_grad_B_alt[:,2,0,0], np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(s.grad_grad_B_alt[:,0,2,0], np.full(nphi, val), rtol=rtol, atol=atol)
                        np.testing.assert_allclose(s.grad_grad_B_alt[:,0,0,2], np.full(nphi, val), rtol=rtol, atol=atol)
                        

class CylindricalCartesianTensorsTests(unittest.TestCase):

    def test_cylindrical_cartesian_tensors(self):
        """
        Test the accuracy and the symmetry in the the cylindrical and Cartesian
        versions of the grad B and grad grad B tensors.
        """
        # Test the grad B tensor in axisymmetry, including symmetry in both indices
        Rmajor = np.random.rand() * 2 + 0.3
        B0 = np.random.rand() * 2 + 1.0
        stel = Qsc(rc=[Rmajor, 0.], zs=[0, 0.], etabar=1.0, B0=B0)
        factor = stel.B0 / Rmajor
        dBdx_cylindrical = stel.grad_B_tensor_cylindrical
        np.testing.assert_almost_equal(dBdx_cylindrical[0, 0], np.zeros(stel.nphi))
        np.testing.assert_almost_equal(dBdx_cylindrical[0, 1], np.full(stel.nphi, -factor))
        np.testing.assert_almost_equal(dBdx_cylindrical[1, 1], np.zeros(stel.nphi))
        np.testing.assert_almost_equal(dBdx_cylindrical, dBdx_cylindrical.transpose(1, 0, 2))
        dBdx_cartesian = stel.grad_B_tensor_cartesian()
        np.testing.assert_almost_equal(dBdx_cartesian[0, 0], np.full(stel.nphi, factor * np.sin(2 * stel.phi)))
        np.testing.assert_almost_equal(dBdx_cartesian[0, 1], np.full(stel.nphi, -factor * np.cos(2 * stel.phi)))
        np.testing.assert_almost_equal(dBdx_cartesian[1, 1], np.full(stel.nphi, -factor * np.sin(2 * stel.phi)))
        np.testing.assert_almost_equal(dBdx_cartesian, dBdx_cartesian.transpose(1, 0, 2))

        # Test the grad B tensor for another configuration, including symmetry in both indices
        stel = Qsc.from_paper(1)
        dBdx_cylindrical = stel.grad_B_tensor_cylindrical
        np.testing.assert_almost_equal(dBdx_cylindrical, dBdx_cylindrical.transpose(1, 0, 2))
        dBdx_cartesian = stel.grad_B_tensor_cartesian()
        np.testing.assert_almost_equal(dBdx_cartesian, dBdx_cartesian.transpose(1, 0, 2))

        # Test the grad grad B tensor, including symmetry in three indices
        dBdxdx_cylindrical = stel.grad_grad_B_tensor_cylindrical()
        np.testing.assert_almost_equal(np.transpose(stel.grad_grad_B, (1, 2, 3, 0)), dBdxdx_cylindrical)
        dBdxdx_cylindrical_transpose_1 = dBdxdx_cylindrical.transpose(0, 2, 1, 3)
        dBdxdx_cylindrical_transpose_2 = dBdxdx_cylindrical.transpose(1, 0, 2, 3)
        dBdxdx_cylindrical_transpose_3 = dBdxdx_cylindrical.transpose(1, 2, 0, 3)
        dBdxdx_cylindrical_transpose_4 = dBdxdx_cylindrical.transpose(2, 0, 1, 3)
        dBdxdx_cylindrical_transpose_5 = dBdxdx_cylindrical.transpose(2, 1, 0, 3)
        np.testing.assert_almost_equal(dBdxdx_cylindrical_transpose_1, dBdxdx_cylindrical)
        np.testing.assert_almost_equal(dBdxdx_cylindrical_transpose_2, dBdxdx_cylindrical)
        np.testing.assert_almost_equal(dBdxdx_cylindrical_transpose_3, dBdxdx_cylindrical)
        np.testing.assert_almost_equal(dBdxdx_cylindrical_transpose_4, dBdxdx_cylindrical)
        np.testing.assert_almost_equal(dBdxdx_cylindrical_transpose_5, dBdxdx_cylindrical)

        dBdxdx_cartesian = stel.grad_grad_B_tensor_cartesian()
        dBdxdx_cartesian_transpose_1 = dBdxdx_cartesian.transpose(0, 2, 1, 3)
        dBdxdx_cartesian_transpose_2 = dBdxdx_cartesian.transpose(1, 0, 2, 3)
        dBdxdx_cartesian_transpose_3 = dBdxdx_cartesian.transpose(1, 2, 0, 3)
        dBdxdx_cartesian_transpose_4 = dBdxdx_cartesian.transpose(2, 0, 1, 3)
        dBdxdx_cartesian_transpose_5 = dBdxdx_cartesian.transpose(2, 1, 0, 3)
        np.testing.assert_almost_equal(dBdxdx_cartesian_transpose_1, dBdxdx_cartesian)
        np.testing.assert_almost_equal(dBdxdx_cartesian_transpose_2, dBdxdx_cartesian)
        np.testing.assert_almost_equal(dBdxdx_cartesian_transpose_3, dBdxdx_cartesian)
        np.testing.assert_almost_equal(dBdxdx_cartesian_transpose_4, dBdxdx_cartesian)
        np.testing.assert_almost_equal(dBdxdx_cartesian_transpose_5, dBdxdx_cartesian)

def test_calculate_grad_B_tensor_vac():
    """Test the calculate_grad_B_tensor_vac method
    """
    stel = Qsc.from_paper("precise QA", I2 = 100, order='r1')
    stel_vac = Qsc.from_paper("precise QA", I2 = 0.0, order='r1')

    # in vacuum, total solution and vacuum components should match
    assert torch.allclose(stel_vac.grad_B_tensor_cylindrical, stel_vac.grad_B_tensor_cylindrical_vac, atol=1e-14), "grad_B_tensor_cylindrical or grad_B_tensor_cylindrical incorect in vacuum"
    assert torch.allclose(stel_vac.grad_B_tensor_cartesian(), stel_vac.grad_B_tensor_cartesian(vacuum_component=True), atol=1e-14), "grad_grad_B or grad_grad_B_vac incorect in vacuum"

    # vacuum component of nonvac field should match the total vacuum solution
    err = stel_vac.grad_B_tensor_cylindrical - stel.grad_B_tensor_cylindrical_vac
    assert torch.allclose(stel.grad_B_tensor_cylindrical_vac, stel_vac.grad_B_tensor_cylindrical, atol=1e-14), "grad_B_tensor_cylindrical or grad_B_tensor_cylindrical_vac incorect"
    assert torch.allclose(stel.grad_B_tensor_cartesian(vacuum_component=True), stel_vac.grad_B_tensor_cartesian(vacuum_component=False), atol=1e-14), "grad_grad_B or grad_grad_B_vac incorect in vacuum"

    print("PASSED: test_calculate_grad_B_tensor_vac")

def test_calculate_grad_grad_B_tensor_vac():
    """Test the calculate_grad_grad_B_tensor_vac method
    """
    stel = Qsc.from_paper("precise QA", I2 = 100, order='r2')
    stel_vac = Qsc.from_paper("precise QA", I2 = 0.0, order='r2')

    # in vacuum, total solution and vacuum components should match
    assert torch.allclose(stel_vac.grad_grad_B, stel_vac.grad_grad_B_vac, atol=1e-14), "grad_grad_B or grad_grad_B_vac incorect in vacuum"
    assert torch.allclose(stel_vac.grad_grad_B_tensor_cartesian(), stel_vac.grad_grad_B_tensor_cartesian(vacuum_component=True), atol=1e-14), "grad_grad_B or grad_grad_B_vac incorect in vacuum"


    # vacuum component of nonvac field should match the total vacuum solution
    err = stel_vac.grad_grad_B - stel.grad_grad_B_vac
    assert torch.allclose(stel.grad_grad_B_vac, stel_vac.grad_grad_B, atol=1e-14), "grad_grad_B or grad_grad_B_vac incorect"
    assert torch.allclose(stel.grad_grad_B_tensor_cartesian(vacuum_component=True), stel_vac.grad_grad_B_tensor_cartesian(vacuum_component=False), atol=1e-14), "grad_grad_B or grad_grad_B_vac incorect in vacuum"


    print("PASSED: test_calculate_grad_grad_B_tensor_vac")

def test_grad_grad_B_tensor_autodiff():
    """
    Test derivativative of grad_B_tensor_cartesian_mse with finite difference.
    """
    # set up the expansion
    stel = Qsc.from_paper("precise QA", order='r2', I2=0.0, p2=0.0)

    # check loss
    grad_grad_B = stel.grad_grad_B_tensor_cartesian()
    target = torch.mean(torch.clone(grad_grad_B)).detach()
    loss = torch.mean((grad_grad_B - target) ** 2)

    dloss_by_ddofs = stel.total_derivative(loss) # list
    dloss_by_drc = dloss_by_ddofs[0]
    dloss_by_dzs = dloss_by_ddofs[1]
    dloss_by_detabar = dloss_by_ddofs[4]

    # check rc gradient with finite difference
    x0 = torch.clone(stel.rc.detach())
    def fd_obj(x):
        stel.rc.data = x
        stel.calculate()
        grad_grad_B = stel.grad_grad_B_tensor_cartesian()
        return torch.mean((grad_grad_B - target) ** 2).detach()
    dloss_by_drc_fd = finite_difference_torch(fd_obj, x0, 1e-9)
    err = torch.max(torch.abs(dloss_by_drc - dloss_by_drc_fd))
    assert err.item() < 1e-3, f"dloss/drc finite difference check failed: {err.item()}"
    stel.rc.data = x0 # restore the original value after finite difference check

    # check zs gradient with finite difference
    x0 = torch.clone(stel.zs.detach())
    def fd_obj(x):
        stel.zs.data = x
        stel.calculate()
        grad_grad_B = stel.grad_grad_B_tensor_cartesian()
        return torch.mean((grad_grad_B - target) ** 2).detach()
    dloss_by_dzs_fd = finite_difference_torch(fd_obj, x0, 1e-9)
    err = torch.max(torch.abs(dloss_by_dzs - dloss_by_dzs_fd))
    assert err.item() < 1e-3, f"dloss/dzs finite difference check failed: {err.item()}"
    stel.zs.data = x0 # restore the original value after finite difference check

    # check etabar gradient with finite difference
    x0 = torch.clone(torch.tensor([stel.etabar.detach()]))
    def fd_obj(x):
        stel.etabar.data = x
        stel.calculate()
        grad_grad_B = stel.grad_grad_B_tensor_cartesian()
        return torch.mean((grad_grad_B - target) ** 2).detach()
    dloss_by_detabar_fd = finite_difference_torch(fd_obj, x0, 1e-4)
    err = torch.abs(dloss_by_detabar - dloss_by_detabar_fd)
    assert err.item() < 1e-5, f"dloss/detabar finite difference check failed: {err.item()}"
    stel.etabar.data = x0 # restore the original value after finite difference check

    print("PASSED: test_grad_grad_B_tensor_autodiff")



if __name__ == "__main__":
    unittest.main()
    test_calculate_grad_B_tensor_vac()
    test_calculate_grad_grad_B_tensor_vac()
    test_grad_grad_B_tensor_autodiff()
