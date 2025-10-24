
import numpy as np
from qsc.qsc import Qsc
import torch
import matplotlib.pyplot as plt
from qsc.util import finite_difference_torch


def test_calculate_r3_vac():

    names = ["precise QH", "precise QA"]
    for name in names:
        stel = Qsc.from_paper(name, I2 = 1.0, p2=-1e5, order='r3')
        stel_vac = Qsc.from_paper(name, I2 = 0.0, p2=0.0, order='r3')

        # in vacuum, total vacuum solution and vacuum components should match
        assert torch.allclose(stel_vac.flux_constraint_coefficient, stel_vac.flux_constraint_coefficient_vac, atol=1e-14)
        assert torch.allclose(stel_vac.B0_order_a_squared_to_cancel, stel_vac.B0_order_a_squared_to_cancel_vac, atol=1e-14)
        assert torch.allclose(stel_vac.d_X3c1_d_varphi, stel_vac.d_X3c1_vac_d_varphi, atol=1e-14)
        assert torch.allclose(stel_vac.d_Y3c1_d_varphi, stel_vac.d_Y3c1_vac_d_varphi, atol=1e-14)
        assert torch.allclose(stel_vac.d_Y3s1_d_varphi, stel_vac.d_Y3s1_vac_d_varphi, atol=1e-14)
        assert torch.allclose(stel_vac.X3c1, stel_vac.X3c1_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Y3c1, stel_vac.Y3c1_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Y3s1, stel_vac.Y3s1_vac, atol=1e-14)
        assert torch.allclose(stel_vac.X3s1, stel_vac.X3s1_vac, atol=1e-14)
        assert torch.allclose(stel_vac.X3s3, stel_vac.X3s3_vac, atol=1e-14)
        assert torch.allclose(stel_vac.X3c3, stel_vac.X3c3_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Y3c3, stel_vac.Y3c3_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Y3s3, stel_vac.Y3s3_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Z3s1, stel_vac.Z3s1_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Z3s3, stel_vac.Z3s3_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Z3c1, stel_vac.Z3c1_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Z3c3, stel_vac.Z3c3_vac, atol=1e-14)
        assert torch.allclose(stel_vac.X3c1_untwisted, stel_vac.X3c1_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Y3c1_untwisted, stel_vac.Y3c1_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Y3s1_untwisted, stel_vac.Y3s1_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.X3s1_untwisted, stel_vac.X3s1_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.X3s3_untwisted, stel_vac.X3s3_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.X3c3_untwisted, stel_vac.X3c3_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Y3c3_untwisted, stel_vac.Y3c3_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Y3s3_untwisted, stel_vac.Y3s3_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Z3s1_untwisted, stel_vac.Z3s1_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Z3s3_untwisted, stel_vac.Z3s3_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Z3c1_untwisted, stel_vac.Z3c1_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Z3c3_untwisted, stel_vac.Z3c3_vac_untwisted, atol=1e-14)

        # vacuum component of nonvac field should match the total vacuum solution
        assert torch.allclose(stel_vac.flux_constraint_coefficient, stel.flux_constraint_coefficient_vac, atol=1e-14)
        assert torch.allclose(stel_vac.B0_order_a_squared_to_cancel, stel.B0_order_a_squared_to_cancel_vac, atol=1e-14)
        assert torch.allclose(stel_vac.d_X3c1_d_varphi, stel.d_X3c1_vac_d_varphi, atol=1e-14)
        assert torch.allclose(stel_vac.d_Y3c1_d_varphi, stel.d_Y3c1_vac_d_varphi, atol=1e-14)
        assert torch.allclose(stel_vac.d_Y3s1_d_varphi, stel.d_Y3s1_vac_d_varphi, atol=1e-14)
        assert torch.allclose(stel_vac.X3c1, stel.X3c1_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Y3c1, stel.Y3c1_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Y3s1, stel.Y3s1_vac, atol=1e-14)
        assert torch.allclose(stel_vac.X3s1, stel.X3s1_vac, atol=1e-14)
        assert torch.allclose(stel_vac.X3s3, stel.X3s3_vac, atol=1e-14)
        assert torch.allclose(stel_vac.X3c3, stel.X3c3_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Y3c3, stel.Y3c3_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Y3s3, stel.Y3s3_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Z3s1, stel.Z3s1_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Z3s3, stel.Z3s3_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Z3c1, stel.Z3c1_vac, atol=1e-14)
        assert torch.allclose(stel_vac.Z3c3, stel.Z3c3_vac, atol=1e-14)
        assert torch.allclose(stel_vac.X3c1_untwisted, stel.X3c1_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Y3c1_untwisted, stel.Y3c1_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Y3s1_untwisted, stel.Y3s1_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.X3s1_untwisted, stel.X3s1_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.X3s3_untwisted, stel.X3s3_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.X3c3_untwisted, stel.X3c3_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Y3c3_untwisted, stel.Y3c3_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Y3s3_untwisted, stel.Y3s3_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Z3s1_untwisted, stel.Z3s1_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Z3s3_untwisted, stel.Z3s3_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Z3c1_untwisted, stel.Z3c1_vac_untwisted, atol=1e-14)
        assert torch.allclose(stel_vac.Z3c3_untwisted, stel.Z3c3_vac_untwisted, atol=1e-14)

    print('PASSED: test_calculate_r3_vac')

if __name__=="__main__":
    test_calculate_r3_vac()