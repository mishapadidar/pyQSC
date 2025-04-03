
import numpy as np
from qsc.qsc import Qsc
import torch
import matplotlib.pyplot as plt
from qsc.util import finite_difference_torch


def test_calculate_r2_vac():

    stel = Qsc.from_paper("precise QA", I2 = 100, order='r2')
    stel_vac = Qsc.from_paper("precise QA", I2 = 0.0, order='r2')

    # in vacuum, total vacuum solution and vacuum components should match
    assert torch.allclose(stel_vac.G2, stel_vac.G2_vac, atol=1e-14)
    assert torch.allclose(stel_vac.d_X20_d_varphi, stel_vac.d_X20_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_X2s_d_varphi, stel_vac.d_X2s_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_X2c_d_varphi, stel_vac.d_X2c_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_Y20_d_varphi, stel_vac.d_Y20_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_Y2s_d_varphi, stel_vac.d_Y2s_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_Y2c_d_varphi, stel_vac.d_Y2c_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_Z20_d_varphi, stel_vac.d_Z20_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_Z2s_d_varphi, stel_vac.d_Z2s_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_Z2c_d_varphi, stel_vac.d_Z2c_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d2_Y1c_d_varphi2, stel_vac.d2_Y1c_vac_d_varphi2, atol=1e-14)
    assert torch.allclose(stel_vac.V1 , stel_vac.V1_vac , atol=1e-14)
    assert torch.allclose(stel_vac.V2 , stel_vac.V2_vac , atol=1e-14)
    assert torch.allclose(stel_vac.V3 , stel_vac.V3_vac , atol=1e-14)
    assert torch.allclose(stel_vac.X20, stel_vac.X20_vac, atol=1e-14)
    assert torch.allclose(stel_vac.X2s, stel_vac.X2s_vac, atol=1e-14)
    assert torch.allclose(stel_vac.X2c, stel_vac.X2c_vac, atol=1e-14)
    assert torch.allclose(stel_vac.Y20, stel_vac.Y20_vac, atol=1e-14)
    assert torch.allclose(stel_vac.Y2s, stel_vac.Y2s_vac, atol=1e-14)
    assert torch.allclose(stel_vac.Y2c, stel_vac.Y2c_vac, atol=1e-14)
    assert torch.allclose(stel_vac.Z20, stel_vac.Z20_vac, atol=1e-14)
    assert torch.allclose(stel_vac.Z2s, stel_vac.Z2s_vac, atol=1e-14)
    assert torch.allclose(stel_vac.Z2c, stel_vac.Z2c_vac, atol=1e-14)
    assert torch.allclose(stel_vac.beta_1s , stel_vac.beta_1s_vac , atol=1e-14)
    assert torch.allclose(stel_vac.B20 , stel_vac.B20_vac , atol=1e-14)
    assert torch.allclose(stel_vac.X20_untwisted, stel_vac.X20_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.X2s_untwisted, stel_vac.X2s_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.X2c_untwisted, stel_vac.X2c_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.Y20_untwisted, stel_vac.Y20_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.Y2s_untwisted, stel_vac.Y2s_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.Y2c_untwisted, stel_vac.Y2c_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.Z20_untwisted, stel_vac.Z20_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.Z2s_untwisted, stel_vac.Z2s_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.Z2c_untwisted, stel_vac.Z2c_vac_untwisted, atol=1e-14)

    # vacuum component of nonvac field should match the total vacuum solution
    assert torch.allclose(stel_vac.G2, stel.G2_vac, atol=1e-14)
    assert torch.allclose(stel_vac.d_X20_d_varphi, stel.d_X20_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_X2s_d_varphi, stel.d_X2s_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_X2c_d_varphi, stel.d_X2c_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_Y20_d_varphi, stel.d_Y20_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_Y2s_d_varphi, stel.d_Y2s_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_Y2c_d_varphi, stel.d_Y2c_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_Z20_d_varphi, stel.d_Z20_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_Z2s_d_varphi, stel.d_Z2s_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d_Z2c_d_varphi, stel.d_Z2c_vac_d_varphi, atol=1e-14)
    assert torch.allclose(stel_vac.d2_Y1c_d_varphi2, stel.d2_Y1c_vac_d_varphi2, atol=1e-14)
    assert torch.allclose(stel_vac.V1 , stel.V1_vac , atol=1e-14)
    assert torch.allclose(stel_vac.V2 , stel.V2_vac , atol=1e-14)
    assert torch.allclose(stel_vac.V3 , stel.V3_vac , atol=1e-14)
    assert torch.allclose(stel_vac.X20, stel.X20_vac, atol=1e-14)
    assert torch.allclose(stel_vac.X2s, stel.X2s_vac, atol=1e-14)
    assert torch.allclose(stel_vac.X2c, stel.X2c_vac, atol=1e-14)
    assert torch.allclose(stel_vac.Y20, stel.Y20_vac, atol=1e-14)
    assert torch.allclose(stel_vac.Y2s, stel.Y2s_vac, atol=1e-14)
    assert torch.allclose(stel_vac.Y2c, stel.Y2c_vac, atol=1e-14)
    assert torch.allclose(stel_vac.Z20, stel.Z20_vac, atol=1e-14)
    assert torch.allclose(stel_vac.Z2s, stel.Z2s_vac, atol=1e-14)
    assert torch.allclose(stel_vac.Z2c, stel.Z2c_vac, atol=1e-14)
    assert torch.allclose(stel_vac.beta_1s , stel.beta_1s_vac , atol=1e-14)
    assert torch.allclose(stel_vac.B20 , stel.B20_vac , atol=1e-14)
    assert torch.allclose(stel_vac.X20_untwisted, stel.X20_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.X2s_untwisted, stel.X2s_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.X2c_untwisted, stel.X2c_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.Y20_untwisted, stel.Y20_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.Y2s_untwisted, stel.Y2s_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.Y2c_untwisted, stel.Y2c_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.Z20_untwisted, stel.Z20_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.Z2s_untwisted, stel.Z2s_vac_untwisted, atol=1e-14)
    assert torch.allclose(stel_vac.Z2c_untwisted, stel.Z2c_vac_untwisted, atol=1e-14)

    print('PASSED: test_calculate_r2_vac')

if __name__=="__main__":
    test_calculate_r2_vac()