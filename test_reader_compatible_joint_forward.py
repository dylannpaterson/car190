"""Numerical equivalence and derivative checks for the new joint convention."""

from types import SimpleNamespace

import numpy as np

from reader_compatible_joint_forward import (
    ReaderCompatiblePhotometryForward, band_terms, internal_reflection,
)


def example_model():
    wave=np.linspace(480.,2300.,81)
    count=3
    r=np.array([-.3+.15j,-.22+.18j,-.35+.09j])[:,None]*(1+.1*np.sin(wave/100))[None,:]
    area=np.array([np.exp(-((wave-650)/150)**2),np.exp(-((wave-1500)/300)**2)])
    return SimpleNamespace(
        clean_reflection_by_tile=r,
        wavelengths_nm=wave,
        effective_area=area,
        star_photon_shapes=np.array([np.ones(len(wave)),(wave/1000)**-.7]),
        tile_thickness_by_id=np.array([
            [[30.,70.,120.],[30.,75.,130.]],
            [[35.,72.,115.],[35.,76.,125.]],
            [[32.,68.,110.],[32.,78.,122.]],
        ]),
        source_tile_ids=np.array([[[0,1,2],[0,2,1]],[[2,0,1],[2,1,0]]]),
        basis=np.array([np.ones(len(wave)),(wave-1400)/1000]),
        mode_count=2,star_count=2,filter_count=2,pass_count=3,tile_count=count,
    )


def test_optical_and_depth_derivatives():
    model=example_model()
    forward=ReaderCompatiblePhotometryForward(model)
    theta=np.array([.01,-.02,.03,.01])
    prediction,jac=forward.evaluate(theta)
    eps=1e-6
    for i in range(len(theta)):
        upper=theta.copy();lower=theta.copy();upper[i]+=eps;lower[i]-=eps
        numerical=(forward.evaluate(upper)[0]-forward.evaluate(lower)[0])/(2*eps)
        np.testing.assert_allclose(jac[...,i],numerical,rtol=1e-5,atol=1e-8)
    wave=model.wavelengths_nm
    thickness=np.array([[50.,100.],[30.,80.],[25.,110.]])
    terms=band_terms(model.clean_reflection_by_tile,wave,thickness)
    upper=band_terms(model.clean_reflection_by_tile,wave,thickness+eps)[0]
    lower=band_terms(model.clean_reflection_by_tile,wave,thickness-eps)[0]
    np.testing.assert_allclose(terms[4],(np.log(upper)-np.log(lower))/(2*eps),rtol=2e-4,atol=1e-8)


def test_forward_matches_reader_algebra_including_changed_tiles():
    model=example_model()
    theta=np.array([.01,-.02,.03,.01])
    actual,_=ReaderCompatiblePhotometryForward(model).evaluate(theta)
    wave=model.wavelengths_nm
    reflection=model.clean_reflection_by_tile*np.exp(theta[:2]@model.basis+1j*(theta[2:]@model.basis))[None,:]
    r,_=internal_reflection(reflection)
    b=-1/9
    from fit_car201_zane_star_reflectance_modulation import trapezoid_weights
    weights=trapezoid_weights(wave)
    expected=np.empty(actual.shape)
    for si in range(2):
        for fi in range(2):
            k=model.star_photon_shapes[si]*model.effective_area[fi]*weights
            for pi in range(3):
                tile=model.source_tile_ids[si,fi,pi]
                d=model.tile_thickness_by_id[tile,fi,pi]
                q=b*r[tile]
                clean=(1-np.abs(r[tile])**2)/np.abs(1+q)**2
                film=np.abs(1+q)**2/np.abs(1+q*np.exp(4j*np.pi*1.25*d/wave))**2
                expected[si,fi,pi]=np.log(np.sum(k*clean*film)/np.sum(k*clean))
    expected-=expected[:,:,:1]
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
