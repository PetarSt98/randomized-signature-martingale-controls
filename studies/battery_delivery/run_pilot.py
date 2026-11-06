"""Run the fixed protocol for a physically volume-limited quarter-hour power call."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
PLAN = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))

def pack(observed):
    """Arrange causal measurements as time, charge, voltage, current and temperature."""
    soc, current, voltage, temperature = np.moveaxis(observed,2,0)
    elapsed = np.broadcast_to(np.linspace(0,1,observed.shape[1]),soc.shape)
    return np.stack((elapsed,soc,voltage,current,temperature),axis=-1)

def unit_call(f):
    """Return normal-model call price, one-trade hedge and residual variance per MWh."""
    gap = f-PLAN['strike']
    sigma = PLAN['settlement_innovation_sd']
    z = gap/sigma
    delta=ndtr(z)
    density = np.exp(-0.5*z*z)/np.sqrt(2*np.pi)
    price=gap*delta+sigma*density
    second=(gap*gap+sigma*sigma)*delta+gap*sigma*density
    irreducible=np.maximum(second-price**2-(sigma*delta)**2,0.)
    return price, delta, irreducible

def run(i):
    from physics import simulate
    from learn import fit_predict
    start=perf_counter()
    directory=OUT/f'split_{i+1}'
    directory.mkdir(exist_ok=True,parents=True)
    batches=[simulate(size,seed) for size,seed in zip([PLAN['train'],PLAN['validation'],PLAN['test']],PLAN['physical_seeds'][i])]
    q_train,q_val,q=[np.asarray(b.volume) for b in batches]
    x_train,x_val,x=[pack(b.observed) for b in batches]
    for volume,m in zip([q_train,q_val,q],[x_train,x_val,x]):
        assert volume.ndim==1 and len(volume)==len(m) and np.isfinite(m).all() and ((volume>=0)&(volume<=1+1e-10)).all()
    del batches
    t_physics=perf_counter()-start
    preds = fit_predict(x_train,q_train,x_val,q_val,x,PLAN['feature_seeds'][i])
    rng=np.random.default_rng(PLAN['financial_seeds'][i])
    f=rng.normal(PLAN['observed_forward_mean'],PLAN['observed_forward_sd'],size=len(q))
    sigma=PLAN['settlement_innovation_sd']
    shock=rng.normal(0,sigma,size=(len(q),PLAN['settlement_simulations_per_history']))
    unit,delta,base=unit_call(f)
    blind=np.repeat(q_train.mean(),len(q))
    preds.update(full_state=q[None,:],mean_volume=blind[None,:])
    premium=blind*unit
    stats={}; per_path={}
    for method,pred in preds.items():
        qhat=np.atleast_2d(pred)
        assert np.isfinite(qhat).all() and qhat.shape[1]==len(q)
        e=qhat-q
        own_quote_mse=(e*unit)**2
        holding=(qhat*delta)
        hedge_specific=(sigma*e*delta)**2
        mse=(q*q*base)+(q*unit-premium)**2+hedge_specific
        payoff=q[:,None]*np.maximum(f[:,None]+shock-PLAN['strike'],0)
        sim_mse=np.mean((payoff[None,:,:]-premium[None,:,None]-holding[:,:,None]*shock[None,:,:])**2,axis=-1)
        cash_mse=q*q*base+own_quote_mse+hedge_specific
        raw=dict(volume_mse=e**2,quote_mse=own_quote_mse,holding_mse=(e*delta)**2,hedge_mse=mse,
                 excess_hedge_mse=hedge_specific,own_quote_and_hedge_mse=cash_mse,
                 realized_hedge_mse=sim_mse,quote_bias=e*unit)
        per_path[method]={k:a.mean(axis=0) for k,a in raw.items()}
        stats[method]={k:float(a.mean()) for k,a in raw.items()}
        stats[method]['draw_volume_mse']=(e**2).mean(axis=1).tolist()
    labels=dict(train_mean=float(q_train.mean()),test_mean=float(q.mean()),test_sd=float(q.std()),
                test_fraction_fully_delivered=float(np.mean(q>=1-1e-8)),
                test_fraction_zero=float(np.mean(q<1e-7)),test_quantiles=np.quantile(q,[0,.1,.25,.5,.75,.9,1]).tolist())
    np.savez_compressed(directory/'predictions.npz',volume=q,forward=f,mean_train_volume=q_train.mean(),
                        **{f'pred_{k}':v for k,v in preds.items() if k not in ('full_state','mean_volume')})
    np.savez_compressed(directory/'paired.npz',**{f'{m}__{k}':a for m,v in per_path.items() for k,a in v.items()})
    (directory/'scores.json').write_text(json.dumps(dict(scores=stats,volume=labels,physics_seconds=t_physics,
            total_seconds=perf_counter()-start,validation=getattr(fit_predict,'last_validation',{})),default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist(),indent=2))
    print(i+1,labels, {m:round(s['volume_mse']**.5,4) for m,s in stats.items()},flush=True)

def summarize():
    dirs=[OUT/f'split_{i+1}' for i in range(PLAN['splits'])]
    records=[json.loads((d/'scores.json').read_text()) for d in dirs]
    keys=records[0]['scores'].keys()
    pooled={}
    for m in keys:
        pooled[m]={k:float(np.mean([r['scores'][m][k] for r in records])) for k in records[0]['scores'][m] if k!='draw_volume_mse'}
        pooled[m]['volume_rmse_by_split']=[r['scores'][m]['volume_mse']**.5 for r in records]
    comparisons={}
    path_batches=[dict(np.load(d/'paired.npz')) for d in dirs]
    for m in keys:
        for other in keys:
            if m==other:continue
            differences={}
            for metric in ['volume_mse','quote_mse','hedge_mse','excess_hedge_mse']:
                v=[s[m+'__'+metric]-s[other+'__'+metric] for s in path_batches]
                dbar=float(np.mean([x.mean() for x in v]))
                se=float(np.sqrt(sum(x.var(ddof=1)/len(x) for x in v))/len(v))
                differences[metric]=dict(mean=dbar,paired_mc_ci=[dbar-1.96*se,dbar+1.96*se],
                    split_means=[float(x.mean()) for x in v])
            comparisons[m+' minus '+other]=differences
    summary=dict(pooled=pooled,paired=comparisons,volume=[r['volume'] for r in records],summed_worker_seconds=sum(r['total_seconds'] for r in records))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    for m,s in pooled.items():
        print(m,*[f'{s[k]**.5:.5f}' for k in ('volume_mse','quote_mse','hedge_mse','realized_hedge_mse','excess_hedge_mse')])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--split", type=int, choices=(1, 2, 3))
    group.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        run(args.split - 1)
