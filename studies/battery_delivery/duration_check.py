"""Supplemental physical check on a three-hour installation with the same contract.
Same SOC, temperature, prior dispatch in C-rates and histories as the high-power case.
Only the number of identical cells available for the prospective 4 MW pulse changes.
"""
import json
from pathlib import Path
import numpy as np
import physics
from run_pilot import PLAN

def check():
    original=physics.P_CELL
    try:
        physics.P_CELL=original*(physics.PACK_MWH/12.)
        volume=np.concatenate([physics.simulate(PLAN['test'],s[2]).volume for s in PLAN['physical_seeds']])
    finally:
        physics.P_CELL=original
    out=dict(duration_hours=3.,n=int(len(volume)),mean_MWh=float(np.mean(volume)),fraction_complete=float(np.mean(volume>=1.-1e-8)),min_MWh=float(np.min(volume)),blind_full_delivery_RMSE_MWh=float(np.sqrt(np.mean((1-volume)**2))))
    path=Path(__file__).resolve().parent/'results'/'three_hour_check.json'
    path.write_text(json.dumps(out,indent=2))
    print(out)

if __name__=='__main__':check()
