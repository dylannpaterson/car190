import numpy as np
import pytest
from per_tile_rate_modulation import PerTileRateModulation


def test_identity_independent_tiles_monotonicity_and_analytic_derivative():
    days=np.array([0.,1.,5.,8.,12.,14.,18.,23.,27.,31.])
    reset=[13.]
    base=np.array([[2.,3.,7.,10.,14.,1.,5.,10.,14.,18.],
                   [1.,2.,4.,7.,11.,2.,4.,8.,10.,12.]])
    model=PerTileRateModulation(days,base,reset,spacing_days=7.)
    zero=np.zeros((2,model.parameter_count))
    np.testing.assert_allclose(model.evaluate(zero),base,atol=1e-12)
    rng=np.random.default_rng(42)
    x=rng.normal(0,.2,zero.shape)
    values,jac=model.evaluate(x,with_jacobian=True)
    for tile in range(2):
        for spec in model.specs:
            assert np.all(np.diff(values[tile,spec["indices"]])>=0)
        for k in range(model.parameter_count):
            plus=x.copy();minus=x.copy();plus[tile,k]+=1e-6;minus[tile,k]-=1e-6
            numerical=(model.evaluate(plus)-model.evaluate(minus))/2e-6
            np.testing.assert_allclose(numerical[tile],jac[tile,:,k],atol=2e-8,rtol=2e-7)
            np.testing.assert_allclose(numerical[1-tile],0.,atol=1e-12)


def test_rejects_decreasing_baseline_within_a_segment():
    with pytest.raises(ValueError,match="decreases"):
        PerTileRateModulation([0.,1.,2.],[[1.,3.,2.]],[])
