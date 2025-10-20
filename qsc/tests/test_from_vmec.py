import torch
import numpy as np

def generate_data_for_test_from_vmec():
    """
    Generate the vmec data needed for the test_from_vmec test.
    """
    from simsopt.mhd.vmec import Vmec
    v = Vmec('./data/input.LandremanPaul2021_QA')
    v.run()

def test_from_vmec():
    """
    Test the from_vmec method of the Qsc class.
    """

    from simsopt.mhd.vmec import Vmec
    from qsc.qsc import Qsc

    """
    First test that we can reproduce the precise QA configuration.
    We compare against the Qsc.from_paper('precise QA') method.
    This configuration is vacuum.
    """

    stel_actual = Qsc.from_paper("precise QA")
    filename = "./input.precise_QA"
    params = {
        "ns_array": [12, 25, 50, 75, 100, 150, 201],
        "niter_array": [1200, 2000, 3000, 4000, 5000, 6000, 20000],
        "ftol_array": [1.0e-17, 1.0e-17, 1.0e-17, 1.0e-17, 1.0e-17, 1.0e-17, 5.0e-16]
    }
    stel_actual.to_vmec(filename, params=params)
    # filename = "./data/wout_precise_QA.nc"
    v = Vmec(filename)
    stel = Qsc.from_vmec(v, n_fourier=len(stel_actual.rc)-1)

    rc_actual = stel_actual.rc.detach().numpy()
    rc = stel.rc.detach().numpy()
    err = np.max(np.abs(rc - rc_actual)) 
    np.testing.assert_almost_equal(rc, rc_actual,
                                   err_msg="rc difference too large", decimal=3)
    
    zs_actual = stel_actual.zs.detach().numpy()
    zs = stel.zs.detach().numpy()
    err = np.max(np.abs(zs - zs_actual))
    np.testing.assert_almost_equal(zs, zs_actual,
                                   err_msg="zs difference too large", decimal=3)
    
    etabar_actual = stel_actual.etabar.detach().numpy()
    etabar = stel.etabar.detach().numpy()
    err = np.max(np.abs(etabar - etabar_actual))
    np.testing.assert_almost_equal(etabar, etabar_actual,
                                   err_msg="etabar difference too large", decimal=1)
    
    B2c_actual = stel_actual.B2c.detach().numpy()
    B2c = stel.B2c.detach().numpy()
    err = np.max(np.abs(B2c - B2c_actual))
    np.testing.assert_almost_equal(B2c, B2c_actual,
                                   err_msg="B2c difference too large", decimal=1)
    
    B2s_actual = stel_actual.B2s.detach().numpy()
    B2s = stel.B2s.detach().numpy()
    err = np.max(np.abs(B2s - B2s_actual))
    np.testing.assert_almost_equal(B2s, B2s_actual,
                                   err_msg="B2s difference too large", decimal=14)
    
    p2_actual = stel_actual.p2.detach().numpy()
    p2 = stel.p2.detach().numpy()
    err = np.max(np.abs(p2 - p2_actual))
    np.testing.assert_almost_equal(p2, p2_actual,
                                   err_msg="p2 difference too large", decimal=1)
    
    I2_actual = stel_actual.I2.detach().numpy()
    I2 = stel.I2.detach().numpy()
    err = np.max(np.abs(I2 - I2_actual))
    np.testing.assert_almost_equal(I2, I2_actual,
                                   err_msg="I2 difference too large", decimal=3)
        
    B0_actual = stel_actual.B0
    B0 = stel.B0
    B0_error = abs(B0 - B0_actual)
    np.testing.assert_almost_equal(B0, B0_actual,
                                  err_msg="B0 difference too large", decimal=3)
    
    spsi_actual = stel_actual.spsi
    spsi = stel.spsi
    spsi_error = abs(spsi - spsi_actual)
    np.testing.assert_almost_equal(spsi, spsi_actual,
                                  err_msg="spsi difference too large", decimal=14)
    
    nfp_actual = stel_actual.nfp
    nfp = stel.nfp
    nfp_error = abs(nfp - nfp_actual)
    np.testing.assert_almost_equal(nfp, nfp_actual,
                                  err_msg="nfp difference too large", decimal=14)
    
    sG_actual = stel_actual.sG
    sG = stel.sG
    sG_error = abs(sG - sG_actual)
    np.testing.assert_almost_equal(sG, sG_actual,
                                  err_msg="sG difference too large", decimal=14)

    # Print max absolute error for sigma0
    iota_actual = stel_actual.iota.detach().numpy()
    iota = stel.iota.detach().numpy()
    iota_error = abs(iota - iota_actual)
    np.testing.assert_almost_equal(iota, iota_actual,
                                  err_msg="iota difference too large", decimal=1)
    
    # Print max absolute error for sigma0
    helicity_actual = stel_actual.helicity.detach().numpy()
    helicity = stel.helicity.detach().numpy()
    helicity_error = abs(helicity - helicity_actual)
    np.testing.assert_almost_equal(helicity, helicity_actual,
                                  err_msg="helicity difference too large", decimal=14)


    """
    Test a QH with pressure and current.
    """
    stel_actual = Qsc.from_paper("precise QH", p2 = -1e2, I2=-1e-6)
    filename = "./input.precise_QH"
    params = {
        "ns_array": [12, 25, 50, 75, 100, 150, 201],
        "niter_array": [1200, 2000, 3000, 4000, 5000, 6000, 20000],
        "ftol_array": [1.0e-17, 1.0e-17, 1.0e-17, 1.0e-17, 1.0e-17, 1.0e-17, 5.0e-16]
    }
    stel_actual.to_vmec(filename, params=params)
    # filename = "./data/wout_precise_QH.nc"
    v = Vmec(filename)
    stel = Qsc.from_vmec(v, n_fourier=len(stel_actual.rc)-1)

    rc_actual = stel_actual.rc.detach().numpy()
    rc = stel.rc.detach().numpy()
    err = np.max(np.abs(rc - rc_actual)) 
    np.testing.assert_almost_equal(rc, rc_actual,
                                   err_msg="rc difference too large", decimal=3)
    
    zs_actual = stel_actual.zs.detach().numpy()
    zs = stel.zs.detach().numpy()
    err = np.max(np.abs(zs - zs_actual))
    np.testing.assert_almost_equal(zs, zs_actual,
                                   err_msg="zs difference too large", decimal=3)
    
    etabar_actual = stel_actual.etabar.detach().numpy()
    etabar = stel.etabar.detach().numpy()
    err = np.max(np.abs(etabar - etabar_actual))
    np.testing.assert_almost_equal(etabar, etabar_actual,
                                   err_msg="etabar difference too large", decimal=1)
    
    B2c_actual = stel_actual.B2c.detach().numpy()
    B2c = stel.B2c.detach().numpy()
    err = np.max(np.abs(B2c - B2c_actual))
    np.testing.assert_almost_equal(B2c, B2c_actual,
                                   err_msg="B2c difference too large", decimal=1)
    
    B2s_actual = stel_actual.B2s.detach().numpy()
    B2s = stel.B2s.detach().numpy()
    err = np.max(np.abs(B2s - B2s_actual))
    np.testing.assert_almost_equal(B2s, B2s_actual,
                                   err_msg="B2s difference too large", decimal=14)
    
    p2_actual = stel_actual.p2.detach().numpy()
    p2 = stel.p2.detach().numpy()
    err = np.max(np.abs(p2 - p2_actual))
    np.testing.assert_almost_equal(p2, p2_actual,
                                   err_msg="p2 difference too large", decimal=1)
    
    I2_actual = stel_actual.I2.detach().numpy()
    I2 = stel.I2.detach().numpy()
    err = np.max(np.abs(I2 - I2_actual))
    np.testing.assert_almost_equal(I2, I2_actual,
                                   err_msg="I2 difference too large", decimal=3)
        
    B0_actual = stel_actual.B0
    B0 = stel.B0
    B0_error = abs(B0 - B0_actual)
    np.testing.assert_almost_equal(B0, B0_actual,
                                  err_msg="B0 difference too large", decimal=3)
    
    spsi_actual = stel_actual.spsi
    spsi = stel.spsi
    spsi_error = abs(spsi - spsi_actual)
    np.testing.assert_almost_equal(spsi, spsi_actual,
                                  err_msg="spsi difference too large", decimal=14)
    
    nfp_actual = stel_actual.nfp
    nfp = stel.nfp
    nfp_error = abs(nfp - nfp_actual)
    np.testing.assert_almost_equal(nfp, nfp_actual,
                                  err_msg="nfp difference too large", decimal=14)
    
    sG_actual = stel_actual.sG
    sG = stel.sG
    sG_error = abs(sG - sG_actual)
    np.testing.assert_almost_equal(sG, sG_actual,
                                  err_msg="sG difference too large", decimal=14)

    # Print max absolute error for sigma0
    iota_actual = stel_actual.iota.detach().numpy()
    iota = stel.iota.detach().numpy()
    iota_error = abs(iota - iota_actual)
    np.testing.assert_almost_equal(iota, iota_actual,
                                  err_msg="iota difference too large", decimal=1)
    
    # Print max absolute error for sigma0
    helicity_actual = stel_actual.helicity.detach().numpy()
    helicity = stel.helicity.detach().numpy()
    helicity_error = abs(helicity - helicity_actual)
    np.testing.assert_almost_equal(helicity, helicity_actual,
                                  err_msg="helicity difference too large", decimal=14)


if __name__ == "__main__":
    # generate_data_for_test_from_vmec()
    test_from_vmec()