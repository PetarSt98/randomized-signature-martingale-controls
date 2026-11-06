"""Dischargeable block energy from a Daigle–Kulkarni electrochemical cell model.

Full equations, sources, units and scenario assumptions are in BATTERY_MODEL.md.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.polynomial import polyval

CHANNELS = ("soc", "current_c", "voltage_v", "temperature_c")
HISTORY_STEPS = 48
DT_MIN = 1
Q_MAX = 13200.0                  # Coulombs, Table 2
CAPACITY_AH = 0.6 * Q_MAX / 3600
R_GAS, FARADAY = 8.314, 96487.
VS, VB = 2.e-6, 2.e-5
FS = VS / (VS+VB)
BLOCK_SECONDS = 900.
REQUEST_MW, PACK_MWH, V_NOM = 4., 2., 3.7
FLEET_CELL_EQUIV = PACK_MWH*1e6/(V_NOM*CAPACITY_AH)
P_CELL = REQUEST_MW*1e6 / FLEET_CELL_EQUIV
LOW_V = 3.
AP = np.array([-33642.23,0.11,23506.89,-74679.26,14359.34,307849.79,85053.13,-1075148.06,2173.62,991586.68,283423.47,-163020.34,-470297.35])
# Algebraic form of the Redlich-Kister correction prevents division by (2*x-1).
POS_COEFF = np.zeros(len(AP)+1)
for k, coeff in enumerate(AP):
    POS_COEFF[k+1] += coeff*(1.+k/2)
    if k: POS_COEFF[k-1] -= coeff*k/2
POS_COEFF /= FARADAY

@dataclass
class BatteryBatch:
    observed: np.ndarray
    volume: np.ndarray

def ocv(soc, temp_c=18.85):
    """Equilibrium cell voltage at uniform SOC, or at the supplied surface SOC."""
    xn=np.clip(.6*soc,1e-7,1.-1e-7)
    xp=1.-xn
    th=R_GAS*(temp_c+273.15)/FARADAY
    return 4.02 + 2.*th*np.log(xn/xp) + polyval(2*xp-1, POS_COEFF) - 86.19/FARADAY*(2*xn-1)

def _voltage(state, t):
    soc, gradient, ohmic, eta = state
    return ocv(soc-gradient/.6,t)-ohmic-eta

def _advance(state, current, seconds, params):
    """Exact constant-current diffusion/ohmic step, midpoint kinetics update."""
    soc, gradient, ohmic, eta = state
    resistance, tau_d, temp = params
    # gradient = mean mole fraction minus surface mole fraction at negative electrode.
    steady=current/Q_MAX*(1/FS-1)*tau_d
    alpha=np.exp(-seconds/tau_d)
    grad_new=steady+(gradient-steady)*alpha
    soc_new=soc-current*seconds/(.6*Q_MAX)
    xn=np.clip(.6*(soc_new+soc)*.5-(grad_new+gradient)*.5,1e-7,1.-1e-7)
    # The two electrodes have equal areas/rate constants and mirror mole fractions.
    kinetic=4*R_GAS*(temp+273.15)/FARADAY*np.arcsinh(current/(8.*np.sqrt(xn*(1-xn))))
    vo_new=ohmic+(resistance*current-ohmic)*(-np.expm1(-seconds/10.))
    eta_new=eta+(kinetic-eta)*(-np.expm1(-seconds/90.))
    return soc_new, grad_new, vo_new, eta_new

def _history_step(state, requested, params):
    # Explicit current limited to an operating window in charge and end-of-minute voltage.
    soc=state[0]
    current=CAPACITY_AH*np.clip(requested,(soc-.92)*60.,(soc-.12)*60.)
    trial=_advance(state,current,60.,params)
    v=_voltage(trial,params[2])
    violates=((v>4.2)&(current<0))|((v<LOW_V)&(current>0))
    if np.any(violates):
        # A one-minute rest replaces an infeasible historical dispatch request.
        current=np.where(violates,0.,current)
        trial=_advance(state,current,60.,params)
    return trial, current/CAPACITY_AH

def deliverable(state, params, step_seconds=5.):
    """Energy at 4 MW until low-voltage cutoff, or 15 minutes, starting now."""
    temp=params[2]
    elapsed=np.zeros_like(state[0])
    active=_voltage(state,temp)>LOW_V
    for _ in range(round(BLOCK_SECONDS/step_seconds)):
        v0=_voltage(state,temp)
        # Constant-power request, using current from beginning-of-substep voltage.
        current=np.where(active,P_CELL/np.maximum(v0,LOW_V),0.)
        trial=_advance(state,current,step_seconds,params)
        v1=_voltage(trial,temp)
        exited=(v1<=LOW_V)|(trial[0]<=0)
        frac=np.clip((v0-LOW_V)/np.maximum(v0-v1,1e-12),0.,1.)
        charge_frac=np.clip(state[0]/np.maximum(state[0]-trial[0],1e-12),0.,1.)
        elapsed+=step_seconds*active*np.where(exited,np.minimum(frac,charge_frac),1.)
        active=active&~exited
        state=trial
    return elapsed/BLOCK_SECONDS   # requested MW * 900s = 1 MWh exactly

def simulate(n, seed, step_seconds=5.):
    rng=np.random.default_rng(seed)
    # Parameter uncertainty is constant for a complete history + prospective block.
    # The published cell is the low-resistance, nominal-diffusion end of this scenario.
    resist=.085*rng.uniform(1.,2.,n)
    diffusion=7e6*rng.uniform(.80,2.,n)
    temp=rng.uniform(15.,25.,n)
    params=resist,diffusion*VS*VB/(VS+VB),temp
    zero=np.zeros(n)
    state=(rng.uniform(.25,.90,n),zero.copy(),zero.copy(),zero.copy())
    x=np.zeros((n,HISTORY_STEPS+1,len(CHANNELS)))
    current=zero.copy()
    for j in range(HISTORY_STEPS+1):
        if j:
            # Exogenous pulses persist for 1/0.22 minutes on average.
            candidate=rng.uniform(-1.,1.,n)
            refresh=(rng.random(n)<.22) if j>1 else np.ones(n,dtype=bool)
            current=np.where(refresh,candidate,current)
            state,actual=_history_step(state,current,params)
        else:
            actual=zero
        x[:,j,:]=np.column_stack((state[0],actual,_voltage(state,temp)+rng.normal(0,.005,n),temp))
    return BatteryBatch(x,deliverable(state,params,step_seconds))
