import numpy as np

import mock_fit_wfi_star_reflectance_modulation as mock
from heatmap_car201_allstar_model_correction import _make_wavelength_grid
from support_aligned_modulation import effective_area_support,configure_basis,gap_basis
from experiment_free_led_modulation import free_value_basis,expand_parameters
from fit_car201_zane_star_reflectance_modulation import RealPhotometryFit


def test_full_support_grid_includes_small_tail_and_zero_brackets(tmp_path):
    wave=np.array([400.,473.,474.,500.,623.,1000.,1670.,2300.,2333.,2334.,2500.])
    values=np.zeros((len(wave),8))
    values[2,0]=.0001
    values[3,0]=1.
    values[7,-1]=1.
    values[8,-1]=.0001
    table=tmp_path/"areas.ecsv"
    lines=[",".join(["Wave",*mock.FILTERS])]
    lines.extend(",".join([str(wavelength/1000),*(str(v) for v in row)]) for wavelength,row in zip(wave,values))
    table.write_text("\n".join(lines)+"\n")
    assert effective_area_support(table)==(474.,2333.)
    grid,area=_make_wavelength_grid(5.,effective_area_path=table,full_effective_area_support=True)
    assert grid[0]<=473. and grid[-1]>=2334.
    assert 474. in grid and 2333. in grid
    assert not area[:,0].any() and not area[:,-1].any()
    assert area[-1,np.flatnonzero(grid==2333.)[0]]>0


def test_support_aligned_endpoint_extension_is_nested_and_keeps_leds_fixed():
    rng=np.random.default_rng(42)
    wave=np.linspace(470,2335,80)
    shape=(2,len(wave))
    model=RealPhotometryFit(wave,np.ones((8,len(wave))),np.ones((3,len(wave))),
        rng.uniform(0,200,(2,8,3)),rng.integers(0,2,(3,8,3)),np.full(shape,.2+.05j),
        np.ones(shape,dtype=complex),np.full(shape,.7),np.ones(2),np.full(shape,4.+.1j),
        np.zeros(2),np.full(2,1.3),"per-gap",entropy_weight=1000.,phase_smoothness_weight=100.,
        pass_zero_point_group_ids=np.array([0,0,1]))
    bounds=(474.,2333.)
    configure_basis(model,bounds)
    parameters=rng.normal(0,.02,model.parameter_count)
    prediction=model.modeled_log_response(parameters[:14])
    penalty=np.sum(model.regularization_residuals(parameters[:14])**2)
    configure_basis(model,bounds,free_endpoints=True)
    expanded=expand_parameters(parameters,7,2)
    np.testing.assert_allclose(model.modeled_log_response(expanded[:18]),prediction,atol=1e-13)
    np.testing.assert_allclose(np.sum(model.regularization_residuals(expanded[:18])**2),penalty,atol=1e-12)
    basis,_=gap_basis(mock.LED_CENTERS_NM,bounds)
    extra,*_=free_value_basis(mock.LED_CENTERS_NM,boundaries=bounds)
    np.testing.assert_allclose(np.concatenate([basis,extra[[0,-1]]]),0.,atol=1e-14)
