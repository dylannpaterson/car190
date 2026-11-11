import numpy as np
from experiment_led_sed_soft_mask import sed_mask


def test_stronger_red_suppression_preserves_blue_and_intervals():
    spectra={i:{"wavelength":np.array([float(i*100),float(i*100+5),float(i*100+10)]),
                "raw_output":np.array([0.,1.,0.])} for i in range(1,7)}
    wave=np.array([102.5,105.,502.5,505.,550.,602.5,605.])
    ordinary=sed_mask(wave,spectra=spectra)
    stronger=sed_mask(wave,spectra=spectra,red_exponent=.25)
    np.testing.assert_allclose(stronger[:2],ordinary[:2])
    assert stronger[2]<ordinary[2] and stronger[5]<ordinary[5]
    assert stronger[3]==0 and stronger[6]==0 and stronger[4]==1
    assert np.all((stronger>=0)&(stronger<=1))
