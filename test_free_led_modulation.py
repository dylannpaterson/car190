import numpy as np

from experiment_free_led_modulation import free_value_basis,augment_basis,expand_parameters
from fit_car201_zane_star_reflectance_modulation import RealPhotometryFit
import mock_fit_wfi_star_reflectance_modulation as mock


def test_free_values_are_independent_at_leds_and_boundaries():
    knots=np.concatenate(([400.],mock.LED_CENTERS_NM,[2321.]))
    basis,returned,weights,_=free_value_basis(knots)
    np.testing.assert_array_equal(returned,knots)
    np.testing.assert_allclose(basis,np.eye(len(knots)),atol=1e-14)
    wave=np.linspace(300,2700,1200)
    basis,*_=free_value_basis(wave)
    np.testing.assert_allclose(basis.sum(axis=0),1.,atol=1e-14)
    np.testing.assert_allclose(weights.sum(),1.)


def test_zero_added_coefficients_preserve_forward_model_and_penalty():
    rng=np.random.default_rng(20261110)
    wave=np.arange(470.,2321.,25.)
    shape=(2,len(wave))
    model=RealPhotometryFit(wave,np.ones((8,len(wave))),np.ones((3,len(wave))),
        rng.uniform(0,200,(2,8,3)),rng.integers(0,2,(3,8,3)),np.full(shape,.2+.05j),
        np.ones(shape,dtype=complex),np.full(shape,.7),np.ones(2),np.full(shape,4.+.1j),
        np.zeros(2),np.full(2,1.3),"per-gap",entropy_weight=1000.,phase_smoothness_weight=100.,
        pass_zero_point_group_ids=np.array([0,0,1]))
    old_modes=model.mode_count
    params=rng.normal(0,.03,model.parameter_count)
    original_prediction=model.modeled_log_response(params[:2*old_modes])
    original_penalty=np.sum(model.regularization_residuals(params[:2*old_modes])**2)
    _,extra=augment_basis(model)
    expanded=expand_parameters(params,old_modes,extra)
    np.testing.assert_allclose(model.modeled_log_response(expanded[:2*model.mode_count]),original_prediction,atol=1e-13)
    np.testing.assert_allclose(np.sum(model.regularization_residuals(expanded[:2*model.mode_count])**2),original_penalty,atol=1e-12)
    assert model.mode_count==15
    assert model.parameter_count==38


def test_endpoint_only_modes_cannot_change_led_values():
    nodes=np.concatenate(([400.],mock.LED_CENTERS_NM,[2321.]))
    basis,*_=free_value_basis(nodes)
    endpoints=basis[[0,-1]]
    np.testing.assert_allclose(endpoints[:,1:-1],0.,atol=1e-14)
    values=np.array([.2,-.1])@endpoints
    np.testing.assert_allclose(values[[0,-1]],[.2,-.1])
    np.testing.assert_allclose(values[1:-1],0.,atol=1e-14)
