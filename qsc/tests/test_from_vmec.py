
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
    # load a vmec equilibria
    from simsopt.mhd.vmec import Vmec
    from qsc.qsc import Qsc
    v = Vmec('./data/wout_LandremanPaul2021_QA.nc')

    Qsc.from_vmec(v, n_fourier=9)

if __name__ == "__main__":
    # generate_data_for_test_from_vmec()
    test_from_vmec()