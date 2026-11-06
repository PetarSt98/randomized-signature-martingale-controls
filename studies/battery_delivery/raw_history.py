"""Supplemental battery forecasts defined after the primary comparison.
Run python raw_history.py --split {1,2,3}.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from learn import _fits, _project, _scale, GAINS, PENALTIES
from physics import simulate
from run_pilot import pack, PLAN, unit_call


def moments(z):
    """Generic first and second moments over time, and moments of first differences."""
    out = [z[:, -1]]
    for a in (z, np.diff(z, axis=1)):
        m = a.mean(axis=1)
        centered = a - m[:, None, :]
        ij = np.triu_indices(a.shape[-1])
        cov = np.einsum('ntj,ntk->njk', centered, centered) / a.shape[1]
        out.extend([m, cov[:, ij[0], ij[1]]])
    return np.column_stack(out)


def run(i):
    index = i-1
    base = Path(__file__).resolve().parent/'results'
    out = base/'extra'/f'split_{i}'
    out.mkdir(parents=True, exist_ok=True)
    batches = [simulate(n, s) for n,s in zip((PLAN['train'],PLAN['validation'],PLAN['test']),
                                            PLAN['physical_seeds'][index])]
    yy = [b.volume for b in batches]
    zz = [pack(b.observed)[:, :, 1:].copy() for b in batches]
    del batches
    seeds = [PLAN['feature_seeds'][index], PLAN['feature_seeds'][index]+100003]
    mu, sd = _scale(zz[0][:,-1])
    snap = [(z[:,-1]-mu)/sd for z in zz]
    fixed = [[np.c_[s, _project(s, seed)] for s in snap] for seed in seeds]
    results, records = {}, {}
    for name in ('raw_history','generic_moments'):
        start = perf_counter()
        u = ([z.reshape(len(z),-1) for z in zz] if name=='raw_history' else [moments(z) for z in zz])
        m,s = _scale(u[0])
        u = [(a-m)/s for a in u]
        losses, fits = [], {}
        def design(j,d,g):
            return np.c_[fixed[d][j], _project(u[j], seeds[d]+40009, g)]
        for g in GAINS:
            diag = []
            for d in range(len(seeds)):
                w,b = _fits(design(0,d,g),yy[0])
                fits[g,d] = w,b
                vpred = np.clip(design(1,d,g)@w+b,0,1)
                diag.append(((vpred-yy[1][:,None])**2).mean(0))
            losses.append(np.mean(diag,axis=0))
        ig,ip = np.unravel_index(np.argmin(losses), (len(GAINS),len(PENALTIES)))
        g=GAINS[ig]
        held = []
        for d in range(len(seeds)):
            w,b=fits[g,d]
            held.append(np.clip(design(2,d,g)@w[:,ip]+b[ip],0,1))
        pred=np.stack(held)
        results['pred_'+name]=pred
        check=np.load(base/f'split_{i}'/'predictions.npz')
        assert np.array_equal(yy[2],check['volume'])
        e=pred-yy[2]
        p,delta,irreduc=unit_call(check['forward'])
        premium=check['mean_train_volume']*p
        variance=PLAN['settlement_innovation_sd']**2
        records[name]=dict(gain=g,ridge=PENALTIES[ip],val_mse=float(losses[ig][ip]),
                           volume_mse=float(np.mean(e**2)),draw_volume_mse=np.mean(e**2,axis=1).tolist(),
                           quote_mse=float(np.mean((e*p)**2)),
                           hedge_mse=float(np.mean(yy[2]**2*irreduc+(yy[2]*p-premium)**2+variance*(e*delta)**2)),
                           seconds=perf_counter()-start)
    np.savez_compressed(out/'predictions.npz',volume=yy[2],**results)
    (out/'scores.json').write_text(json.dumps(records,indent=2))
    print('split',i,records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--split',type=int,choices=(1,2,3),required=True)
    run(p.parse_args().split)
