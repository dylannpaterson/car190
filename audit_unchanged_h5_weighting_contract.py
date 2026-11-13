"""Check which source clean-response information the unchanged reader can express."""

import json

import h5py
import numpy as np

import compare_eric_v3_amber_car201 as c
from package_joint_h5_unchanged_contract import DEFAULT_SOURCE, DEFAULT_OUTPUT


def main():
    records=[]
    with c.load_eric_v3_reader(c.DEFAULT_SOURCE_ZIP) as reader:
        model=reader.IceResponseModel(DEFAULT_OUTPUT)
        original_offsets=model._d_off.copy()
        try:
            # Controlled zero-ice test of the public API: no code change and no
            # fitted model change is persisted. This probes its normalization.
            model._d_off[:]=0.
            with h5py.File(DEFAULT_SOURCE) as source:
                pg=source['photometry']
                mapping=pg['superpixel_to_tile']
                native=pg['wavelength_nm'][()]
                wave=np.asarray(model.wavelength_nm,float)
                for detector in ('WFI01','WFI03','WFI06'):
                    area_path,_=c.amber.canonical_configuration(detector)
                    areas,_=c.effective_area_on_model_grid(detector,area_path,wave)
                    quadrature=c.workflow.fit_model.trapezoid_weights(wave)
                    # A flat photon spectrum is sufficient as a counterexample:
                    # a conversion valid for arbitrary sources must preserve it.
                    weight=areas[0]*quadrature
                    tiles=np.array([0,136,255])
                    x=(tiles%16)*256+128
                    y=(tiles//16)*256+128
                    sp=model.superpixel(int(detector[3:]),x,y)
                    np.testing.assert_allclose(model.thickness(int(detector[3:]),x,y,[0.]),0.,atol=1e-12)
                    public=model.band_response(
                        int(detector[3:]),x,y,[0.],wave,weight,
                        gray=True,no_ice_weight=True,
                    )[:,0]
                    np.testing.assert_allclose(public,1.,atol=1e-12)
                    for tile,s,value in zip(tiles,sp,public):
                        curve=pg['clean_relative_weight'][int(mapping[int(s)])]
                        curve=np.interp(wave,native,curve)
                        fitted_clean_ratio=float(np.sum(weight*curve)/np.sum(weight))
                        records.append(dict(
                            detector=detector,tile=int(tile),filter='F062',
                            source_clean_modulated_over_unmodulated=fitted_clean_ratio,
                            unchanged_reader_zero_ice_band_response=float(value),
                        ))
        finally:
            model._d_off[:]=original_offsets
    report=dict(
        test='zero ice, flat photon spectrum, detector-specific F062 effective area',
        result='Source clean-state ratio is not representable by the unchanged normalized response at zero ice.',
        records=records,
        no_model_refitting=True,no_reader_edits=True,
    )
    output=DEFAULT_OUTPUT.parent/'weighting_contract_audit.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
