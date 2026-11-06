"""Numerical checks for the battery delivery study. Run with pytest or as a script."""
import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "battery_delivery"))

import physics as p
from run_pilot import PLAN, unit_call


def test_equilibrium_and_constant_load():
    # Independent RK evaluation of equation 10, separately for both electrodes.
    s=np.array([.10,.30,.50,.70,.90])
    xn=.6*s; xp=1.-xn
    def correction(x, a):
        y=2*x-1
        return sum(c*(y**(k+1)-(2*x*k*(1-x)*y**(k-1) if k else 0.)) for k,c in enumerate(a))/p.FARADAY
    rt=p.R_GAS*292./p.FARADAY
    independent=4.02+rt*np.log((1-xp)/xp)-rt*np.log((1-xn)/xn)
    independent+=correction(xp,p.AP)-correction(xn,[86.19])
    assert_allclose(p.ocv(s,18.85),independent,atol=1e-12)
    temp=np.array([18.85]); diffusion=7e6
    params=(np.array([.085]),np.array([diffusion*p.VS*p.VB/(p.VS+p.VB)]),temp)
    zero=np.array([0.]);init=np.array([.64]);state=(init,zero,zero,zero)
    current=1.4
    for _ in range(10): state=p._advance(state,current,60.,params)
    # Independent published equations for concentrations, ohmic drop and activation
    # (four ion compartments and separate electrode kinetics) via solve_ivp.
    from scipy.integrate import solve_ivp
    xn0=.6*init[0];xp0=1-xn0
    y0=[p.Q_MAX*(1-p.FS)*xn0,p.Q_MAX*p.FS*xn0,p.Q_MAX*(1-p.FS)*xp0,p.Q_MAX*p.FS*xp0,0.,0.,0.]
    def deriv(t,y):
        qnb,qns,qpb,qps,oh,en,ep=y
        jn=(qnb/p.VB-qns/p.VS)/diffusion
        jp=(qpb/p.VB-qps/p.VS)/diffusion
        norm=2*p.R_GAS*292/p.FARADAY
        def eta(z): return norm*np.arcsinh(current/(8*np.sqrt(z*(1-z))))
        return [-jn,jn-current,-jp,jp+current,(.085*current-oh)/10,
                (eta(qns/(p.FS*p.Q_MAX))-en)/90,(eta(qps/(p.FS*p.Q_MAX))-ep)/90]
    y=solve_ivp(deriv,[0,600],y0,rtol=1e-9,atol=1e-10).y[:,-1]
    assert_allclose(state[0],(y[0]+y[1])/(.6*p.Q_MAX),atol=1e-10)
    assert_allclose(.6*state[0]-state[1], y[1]/(p.FS*p.Q_MAX),atol=1e-9)
    assert_allclose(y[2]+y[3]+y[0]+y[1],p.Q_MAX,atol=1e-7)
    v=p.ocv(y[1]/(.6*p.FS*p.Q_MAX),18.85)-sum(y[4:])
    assert_allclose(p._voltage(state,temp),v,atol=3e-5)
    print('constant 600s discharge, voltage difference vs independent ODE solver (V):',
          float((p._voltage(state,temp)-v).item()))

def test_paths():
    b=p.simulate(3000,411,5);fine=p.simulate(3000,411,1)
    assert_array_equal(b.observed,fine.observed)
    assert b.observed.shape==(3000,49,4)
    assert np.isfinite(b.observed).all() and np.isfinite(b.volume).all()
    assert np.all((b.volume>=0)&(b.volume<=1))
    assert_allclose(np.diff(b.observed[:,:,0],axis=1),-b.observed[:,1:,1]/60,atol=1e-12)
    assert_allclose(np.diff(b.observed[:,:,3],axis=1),0)
    diff=b.volume-fine.volume
    print('5s vs 1s MWh cutoff solver:',dict(bias=diff.mean(),rmse=np.sqrt(np.mean(diff**2)),
                                                p95_abs=np.quantile(abs(diff),.95),max_abs=max(abs(diff))))
    assert np.sqrt(np.mean(diff**2))<.003
    print('volume percentiles [0,5,25,50,75,95,100], MWh:',np.percentile(b.volume,[0,5,25,50,75,95,100]))

def test_call_price_and_hedging_error():
    k=PLAN['strike']; s=PLAN['settlement_innovation_sd']
    for f in [58.,80.,102.]:
        p,d,r=unit_call(f)
        cut=(k-f)/s
        ref=quad(lambda z:(f+s*z-k)*norm.pdf(z),cut,np.inf)[0]
        cov=quad(lambda z:s*z*(f+s*z-k)*norm.pdf(z),cut,np.inf)[0]
        assert_allclose(p,ref,rtol=1.e-10)
        assert_allclose(d,cov/s**2,rtol=1.e-10)
        q=.38; qhat=.29; common=.24*p
        def residual(z):
            return (q*max(f+s*z-k,0.)-common-qhat*d*s*z)**2*norm.pdf(z)
        direct=quad(residual,-np.inf,cut)[0]+quad(residual,cut,np.inf)[0]
        formula=q*q*r+(q*p-common)**2+(s*(qhat-q)*d)**2
        assert_allclose(direct,formula,rtol=1.e-10)
    print('Premium, minimum-variance hedge and settlement error pass quadrature checks at 3 forward prices.')


if __name__ == "__main__":
    test_equilibrium_and_constant_load()
    test_paths()
    test_call_price_and_hedging_error()
