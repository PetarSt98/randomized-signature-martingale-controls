"""Exploratory follow-up with a fixed protocol and wholly independent confirmation paths.
All feature budgets compare mean squared error over draws. Four scalable baselines
and three possible additions to a fixed 64-random-column moments baseline.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from learn import _fits, _project, _scale, _encode, _summaries, GAINS, PENALTIES
from physics import simulate
from run_pilot import pack, unit_call, PLAN
from raw_history import moments

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "confirmation_protocol.json").read_text(encoding="utf-8"))
if PROTOCOL["gains"] != list(GAINS) or PROTOCOL["ridges"] != list(PENALTIES):
    raise RuntimeError("Feature gains or ridge penalties differ from the confirmation protocol.")


def run(i):
    out=ROOT/'results'/'confirmation'/f'split_{i}'
    out.mkdir(parents=True,exist_ok=True)
    part=i-1
    sam=[simulate(n,s) for n,s in zip(PROTOCOL['sizes'],PROTOCOL['phys'][part])]
    yy=[v.volume for v in sam]; xx=[pack(v.observed) for v in sam]
    del sam
    # The financial scenarios are generated for scoring only.
    f=np.random.default_rng(PROTOCOL['finance'][part]).normal(PLAN['observed_forward_mean'],PLAN['observed_forward_sd'],len(yy[2]))
    unit,delta,irreduc=unit_call(f); sigma=PLAN['settlement_innovation_sd']
    pblind=yy[0].mean()*unit
    seeds=[PROTOCOL['features'][part],PROTOCOL['features'][part]+100003]
    state=[a[:,-1,1:] for a in xx]
    def norm(u):
        m,s=_scale(u[0]);return [(a-m)/s for a in u]
    snap=norm(state)
    u=dict(static=snap, moments=norm([moments(a[:,:,1:]) for a in xx]))
    for name in ('ewma','circuit'):u[name]=norm([_summaries(a,name) for a in xx])
    scale=_scale(xx[0][:,:,1:],axes=(0,1))
    shared=[[np.c_[s,_project(s,seed)] for s in snap] for seed in seeds]
    cache={}
    def enc(j,d,g):
        key=(j,d,g)
        if key not in cache: cache[key]=_encode(xx[j], seeds[d]+40009,g,scale)
        return cache[key]
    rec,allpred={},{}
    for name in PROTOCOL['methods']:
        start=perf_counter()
        addon=name.startswith('moments_plus')
        def design(j,d,g):
            if name in ('reservoir','moments_plus_reservoir'):
                v=enc(j,d,g)
            else:
                src={'moments_plus_snapshot':'static','moments_plus_moments':'moments'}.get(name,name)
                v=_project(u[src][j],seeds[d]+(72391 if addon else 40009),g)
            prefix=base[d][j] if addon else shared[d][j]
            return np.c_[prefix,v]
        loss,fits=[],{}
        for g in GAINS:
            draw=[]
            for d in range(2):
                w,b=_fits(design(0,d,g),yy[0]);fits[g,d]=(w,b)
                pq=np.clip(design(1,d,g)@w+b,0,1)
                draw.append(((pq-yy[1][:,None])**2).mean(0))
            loss.append(np.mean(draw,axis=0))
        ig,ip=np.unravel_index(np.argmin(loss),(len(GAINS),len(PENALTIES)))
        g=GAINS[ig]; vmse=float(loss[ig][ip]); decision='model'
        if addon and vmse>=rec['moments']['val_mse']:
            pred=allpred['moments'];vmse=rec['moments']['val_mse'];decision='revert'
        else:
            pp=[]
            for d in range(2):
                w,b=fits[g,d]
                pp.append(np.clip(design(2,d,g)@w[:,ip]+b[ip],0,1))
            pred=np.stack(pp)
        if name=='moments':
            base=[[design(j,d,g) for j in range(3)] for d in range(2)]
        allpred[name]=pred
        q=yy[2];err=pred-q
        rec[name]=dict(gain=g,ridge=PENALTIES[ip],val_mse=vmse,choice=decision,
                       volume_mse=float(np.mean(err**2)),quote_mse=float(np.mean((unit*err)**2)),
                       hedge_mse=float(np.mean(q*q*irreduc+(q*unit-pblind)**2+(sigma*delta*err)**2)),
                       elapsed=perf_counter()-start)
    np.savez_compressed(out/'predictions.npz',volume=yy[2],forward=f,mean_train_volume=yy[0].mean(),**{'pred_'+k:v for k,v in allpred.items()})
    (out/'scores.json').write_text(json.dumps(rec,indent=2))
    print(i,{k:round(v['volume_mse']**.5,5) for k,v in rec.items()},flush=True)


def summary():
    dst=ROOT/'results'/'confirmation'
    info=[json.loads((dst/f'split_{j}'/'scores.json').read_text()) for j in (1,2,3)]
    res={m:{k:float(np.mean([r[m][k] for r in info])) for k in ('volume_mse','quote_mse','hedge_mse')} for m in PROTOCOL['methods']}
    acc={}
    rs=[dict(np.load(dst/f'split_{j}'/'predictions.npz')) for j in (1,2,3)]
    for alt in PROTOCOL['methods']:
        if alt=='moments_plus_reservoir':continue
        out={}
        for metric in ('volume_mse','quote_mse','hedge_mse'):
            ds=[]
            for a in rs:
                q=a['volume'];f=a['forward'];unit,delta,_=unit_call(f)
                diff=np.mean((a['pred_moments_plus_reservoir']-q)**2-(a['pred_'+alt]-q)**2,axis=0)
                w={'volume_mse':1.,'quote_mse':unit**2,'hedge_mse':(PLAN['settlement_innovation_sd']*delta)**2}[metric]
                ds.append(w*diff)
            avg=np.mean([d.mean() for d in ds]);se=np.sqrt(sum(d.var(ddof=1)/len(d) for d in ds))/3
            out[metric]={'split_differences':[d.mean() for d in ds],'mean':avg,'MC_interval':[avg-1.96*se,avg+1.96*se]}
        acc[alt]=out
    (dst/'summary.json').write_text(json.dumps({'pooled':res,'moments_plus_reservoir_minus':acc},indent=2))
    for n,x in res.items():print(n, {k:round(v**.5,6) for k,v in x.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--split", type=int, choices=(1, 2, 3))
    group.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary()
    else:
        run(args.split)
