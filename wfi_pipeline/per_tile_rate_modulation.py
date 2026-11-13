"""Positive independent growth-rate modulations around fixed tile histories."""

import numpy as np


class PerTileRateModulation:
    """Integrate positive adjusted growth increments on a fixed common time grid.

    Each tile/segment has an independent initial-thickness factor and linearly
    interpolated log-rate multipliers, spaced no more than ``spacing_days``.
    Growth increments remain nonnegative and reset boundaries remain separate.
    """
    def __init__(self,days,baseline_thickness,reset_days,spacing_days=7.):
        self.days=np.asarray(days,float)
        self.baseline=np.asarray(baseline_thickness,float)
        self.resets=np.sort(np.asarray(reset_days,float))
        if self.baseline.ndim!=2 or self.baseline.shape[1]!=len(self.days):
            raise ValueError("Baseline must have shape (tile,time)")
        if np.any(np.diff(self.days)<=0) or spacing_days<=0:
            raise ValueError("Times must increase strictly; spacing must be positive")
        if not np.isfinite(self.baseline).all() or np.any(self.baseline<0):
            raise ValueError("Baseline thickness must be finite and nonnegative")
        self.segments=np.searchsorted(self.resets,self.days,side="right")
        self.specs=[];self.parameter_count=0
        for segment in range(len(self.resets)+1):
            indices=np.flatnonzero(self.segments==segment)
            if len(indices)<2:raise ValueError("Each segment needs at least two observations")
            local=self.days[indices]
            increments=np.diff(self.baseline[:,indices],axis=1)
            if np.any(increments < -1e-8):raise ValueError("Baseline decreases inside an ice-growth segment")
            increments=np.maximum(increments,0.)
            count=max(2,int(np.ceil((local[-1]-local[0])/spacing_days))+1)
            knots=np.linspace(local[0],local[-1],count)
            midpoint=(local[1:]+local[:-1])/2
            basis=np.column_stack([np.interp(midpoint,knots,np.eye(count)[i]) for i in range(count)])
            start=self.parameter_count
            self.parameter_count+=1+count
            self.specs.append({"segment":segment,"indices":indices,"knots":knots,"basis":basis,
                               "increments":increments,"initial_index":start,"rate_slice":slice(start+1,start+1+count)})

    def evaluate(self,parameters,with_jacobian=False):
        parameters=np.asarray(parameters,float)
        if parameters.shape!=(self.baseline.shape[0],self.parameter_count):
            raise ValueError("Incorrect per-tile trajectory parameter shape")
        result=np.empty_like(self.baseline)
        jac=np.zeros((*result.shape,self.parameter_count)) if with_jacobian else None
        for spec in self.specs:
            idx=spec["indices"]; initial_index=spec["initial_index"]; rate_slice=spec["rate_slice"]
            initial=self.baseline[:,idx[0]]*np.exp(parameters[:,initial_index])
            rate_log=parameters[:,rate_slice]@spec["basis"].T
            increments=spec["increments"]*np.exp(rate_log)
            result[:,idx]=initial[:,None]+np.column_stack((np.zeros(len(result)),np.cumsum(increments,axis=1)))
            if with_jacobian:
                jac[:,idx,initial_index]=initial[:,None]
                derivatives=increments[:,:,None]*spec["basis"][None,:,:]
                jac[:,idx,rate_slice]=np.concatenate((np.zeros((len(result),1,derivatives.shape[-1])),
                                                       np.cumsum(derivatives,axis=1)),axis=1)
        return (result,jac) if with_jacobian else result
