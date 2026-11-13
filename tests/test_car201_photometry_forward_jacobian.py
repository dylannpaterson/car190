import numpy as np

from car201_photometry_forward_jacobian import CachedPhotometryForward
from fit_car201_zane_star_reflectance_modulation import RealPhotometryFit


def test_cached_response_and_analytic_derivatives_match_original_forward_model():
    rng = np.random.default_rng(20261110)
    wave = np.arange(470., 2321., 25.)
    tiles, stars, bands, passes = 2, 3, 8, 3
    shape = (tiles,len(wave))
    model = RealPhotometryFit(
        wave, rng.uniform(.1,1.,(bands,len(wave))), rng.uniform(.5,2.,(stars,len(wave))),
        rng.uniform(0.,200.,(tiles,bands,passes)),
        rng.integers(0,tiles,(stars,bands,passes)),
        np.full(shape,.2+.05j), np.full(shape,1.+.1j), np.full(shape,.7),
        np.ones(tiles),np.full(shape,4.+.1j),np.zeros(tiles),np.full(tiles,1.3),"per-gap",
    )
    parameters = rng.normal(0,.03,2*model.mode_count)
    cached = CachedPhotometryForward(model)
    value,jacobian = cached.evaluate(parameters)
    np.testing.assert_allclose(value,model.modeled_log_response(parameters),atol=1e-13,rtol=1e-11)
    for index in range(len(parameters)):
        upper,lower=parameters.copy(),parameters.copy()
        upper[index]+=1e-6
        lower[index]-=1e-6
        derivative=(model.modeled_log_response(upper)-model.modeled_log_response(lower))/(2e-6)
        np.testing.assert_allclose(jacobian[:,:,:,index],derivative,atol=1e-8,rtol=1e-4)
