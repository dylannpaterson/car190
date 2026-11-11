"""Detector-specific modulation boundaries derived from imaging effective area."""

import numpy as np

import mock_fit_wfi_star_reflectance_modulation as mock
from experiment_free_led_modulation import free_value_basis


def effective_area_support(path):
    wave,area=mock.load_effective_area(path)
    indices=np.flatnonzero(np.any(area>0.,axis=0))
    if not len(indices) or indices[0]==0 or indices[-1]==len(wave)-1:
        raise ValueError("Imaging throughput must be bracketed by zero values")
    bounds=(float(wave[indices[0]]),float(wave[indices[-1]]))
    if not bounds[0]<min(mock.LED_CENTERS_NM)<max(mock.LED_CENTERS_NM)<bounds[1]:
        raise ValueError("Effective-area support does not contain all six LEDs")
    return bounds


def gap_basis(wave,bounds):
    wave=np.asarray(wave,float)
    knots=np.concatenate(([bounds[0]],mock.LED_CENTERS_NM,[bounds[1]]))
    basis=[]
    for lower,upper in zip(knots[:-1],knots[1:]):
        values=np.zeros_like(wave)
        mask=(wave>=lower)&(wave<=upper)
        values[mask]=np.sin(np.pi*(wave[mask]-lower)/(upper-lower))**2
        basis.append(values)
    return np.asarray(basis),knots


def configure_basis(model,bounds,free_endpoints=False):
    basis,knots=gap_basis(model.wavelengths_nm,bounds)
    widths=np.diff(knots)
    count=len(basis)
    model.basis=basis
    model.basis_names=[f"gap_{left:.6f}_{right:.6f}nm" for left,right in zip(knots[:-1],knots[1:])]
    model.basis_centers_nm=(knots[:-1]+knots[1:])/2
    model.node_weights=widths/widths.sum()
    model.node_groups=[np.asarray([i]) for i in range(count)]
    model.group_bounds=list(zip(knots[:-1],knots[1:]))
    if free_endpoints:
        extra,nodes,weights,support=free_value_basis(model.wavelengths_nm,boundaries=bounds)
        selected=[0,len(nodes)-1]
        model.basis=np.concatenate([model.basis,extra[selected]])
        model.basis_names+=[f"free_value_{nodes[index]:.6f}nm" for index in selected]
        model.basis_centers_nm=np.concatenate([model.basis_centers_nm,nodes[selected]])
        model.node_weights=np.concatenate([model.node_weights,weights[selected]])
        model.node_groups+=[np.asarray([count+i]) for i in range(2)]
        model.group_bounds+=[support[index] for index in selected]
    model.mode_count=len(model.basis)
    model.basis_grouping="support-aligned-gaps"+("-free-endpoints" if free_endpoints else "")
    model.modulation_boundaries_nm=tuple(bounds)
    return count
