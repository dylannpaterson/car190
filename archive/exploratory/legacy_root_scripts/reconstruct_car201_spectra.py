"""Reconstruct observer-frame posterior spectrum draws from the shared grid."""
import h5py,numpy as np
from itertools import product
from extinction import fitzpatrick99

def spectra_for_source(source_id, posterior_path,grid_path, samples=None):
 """Return wavelength [um], F_lambda draws [erg/s/cm2/um]."""
 with h5py.File(posterior_path,'r') as h:
  ids=h['source_id'][:].astype('U'); hits=np.flatnonzero(ids==str(source_id))
  if len(hits)!=1:raise KeyError(f'Expected one source, got {len(hits)}')
  i=int(hits[0]);
  if h['state'][i]!=1:raise ValueError('Source fit did not succeed')
  theta=h['samples'][i];theta=theta if samples is None else theta[samples]
 with h5py.File(grid_path,'r') as h:
  wave=h['wavelength_um'][:].astype(float); p=h['grid_parameters'][:]; axis=[np.unique(p[:,j]) for j in range(3)];lookup={tuple(row):j for j,row in enumerate(p)}; result=[];cache={}
  for teff,logg,feh,norm,av in theta:
   coords=(float(logg),float(teff),float(feh)); brackets=[]
   for a,v in zip(axis,coords):
    k=np.searchsorted(a,v)
    if k==0:lo=hi=0;frac=0.
    elif k==len(a):lo=hi=len(a)-1;frac=0.
    elif np.isclose(v,a[k],rtol=0,atol=1e-9):lo=hi=k;frac=0.
    else:lo=k-1;hi=k;frac=(v-a[lo])/(a[hi]-a[lo])
    brackets.append((lo,hi,frac))
   flux=np.zeros_like(wave)
   for bits in product((0,1),repeat=3):
    wt=np.prod([b[2] if bit else 1-b[2] for b,bit in zip(brackets,bits)])
    if wt==0:continue
    key=tuple(float(a[b[bit]]) for a,b,bit in zip(axis,brackets,bits));j=lookup.get(key)
    if j is None:raise ValueError(f'Grid interpolation corner absent: {key}')
    if j not in cache:cache[j]=h['surface_flux'][j].astype(float)
    flux+=wt*cache[j]
   attenuation=10**(-.4*fitzpatrick99(wave*1e4,float(av),3.1));result.append(flux*float(norm)*attenuation)
 return wave,np.array(result)
