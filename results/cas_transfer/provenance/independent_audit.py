"""Independent saved-artifact verification: no fitting or reporting imports."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/cas_transfer'
r = json.loads((OUT / 'results.json').read_text())
p = json.loads((OUT / 'plan.json').read_text())
assert r['status'] == 'completed'
assert len(r['fits']) == 360 and len(r['evaluations']) == 132
assert sum(len(e['roles']) for e in r['evaluations'].values()) == 186
assert len(r['datasets']) == 12 and not r['failures']
assert all(f['status'] == 'completed' for f in r['fits'].values())
for name, digest in r['source_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
assert hashlib.sha256((ROOT/'results/cas_costs/results.json').read_bytes()).hexdigest() == r['reference_results_sha256']

def equal(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)

# Reconstruct independent simulator partitions and manually discounted payoffs.
from rsmc.black_scholes import BlackScholesModel
from rsmc.heston import HestonModel
from rsmc.multi_asset_heston import MultiAssetHestonModel
models = {'black_scholes':BlackScholesModel,'heston':HestonModel,'basket':MultiAssetHestonModel}
test_payoffs = {}
for key, d in r['datasets'].items():
    model = models[d['task']](**d['model'])
    seeds = np.random.SeedSequence(d['dataset_seed']).spawn(3)
    for (split, size), seed in zip(d['sizes'].items(), seeds):
        paths = model.simulate(size, d['n_steps'], np.random.default_rng(seed))
        assert hashlib.sha256(paths.spot.tobytes()).hexdigest() == d['dataset_spot_sha256'][split]
        prices = paths.spot[:,1:]
        if d['task'] == 'basket':
            prices = prices.mean(axis=2)
        payoff = np.exp(-model.r*paths.times[-1])*np.maximum(prices.mean(axis=1)-d['strike'],0.)
        if split == 'train':
            equal(float(np.float32(payoff.mean())),d['premium'])
        if split == 'test':
            test_payoffs[key] = payoff

# Independent checkpoint and six-tuple validation selection.
for f in r['fits'].values():
    assert f['premium'] == r['datasets'][f['dataset_id']]['premium']
    if f.get('history'):
        scores = [h['validation_mse'] for h in f['history']]
        best = int(np.argmin(scores))
        equal(f['validation_mse'],scores[best])
        if f['method'] == 'transformer':
            assert f['best_epoch'] == f['history'][best]['epoch'] <= 20
        else:
            assert f['best_step'] == f['history'][best]['step'] <= 120
for task in p['tasks']:
    for condition in p['conditions']:
        means=[]
        for index in range(6):
            draw_scores=[]
            for seed in p['draw_seeds']['tuning']:
                fits=[f for f in r['fits'].values() if f['phase']=='tuning' and f['task']==task and f['condition']==condition and f['candidate_index']==index and f['seed']==seed]
                assert len(fits) == (1 if condition=='zero' else 2)
                draw_scores.append(min(f['validation_mse'] for f in fits))
            means.append(np.mean(draw_scores))
        assert int(np.argmin(means)) == r['selection'][task][condition]['candidate_index'] == 0
        equal(r['selection'][task][condition]['validation_mse'],means[0])

metrics_by_id={}
seen=set()
for key,e in r['evaluations'].items():
    with np.load(OUT/e['array_file']) as saved:
        z={k:saved[k] for k in saved.files}
    assert all(v.shape==(8000,) and np.isfinite(v).all() for v in z.values())
    equal(z['payoff'],test_payoffs[e['dataset_id']])
    err=z['payoff']-e['premium']-z['gain']+z['cost']
    equal(z['errors'],err)
    equal(z['cost'],z['proportional_cost']+z['temporary_cost'])
    assert np.all(z['proportional_cost']>=0) and np.all(z['temporary_cost']>=0)
    assert np.all(z['share_turnover']>=0) and np.all(z['notional_turnover']>=0)
    equal(z['proportional_cost'],p['conditions'][e['condition']]['kappa']*z['notional_turnover'])
    if e['condition']=='zero':
        assert not np.count_nonzero(z['cost'])
    m={'net_mse':np.mean(err**2),'net_rmse':np.sqrt(np.mean(err**2)),
       'mean_error':np.mean(err),'mean_error_se':np.std(err,ddof=1)/np.sqrt(len(err)),
       'mean_gain':np.mean(z['gain']), 'mean_turnover':np.mean(z['share_turnover']),
       'mean_notional_turnover':np.mean(z['notional_turnover']),
       'mean_proportional_cost':np.mean(z['proportional_cost']),
       'mean_temporary_cost':np.mean(z['temporary_cost']), 'mean_cost':np.mean(z['cost']),
       'premium':e['premium'],'n_eval':len(err)}
    for name,value in m.items():
        equal(e['metrics'][name],value)
    metrics_by_id[key]=m
    identity=(e['dataset_id'],e['condition'],e['candidate_index'],e['seed'])
    assert identity not in seen
    seen.add(identity)
    fitted=r['fits'][e['fit_id']]
    assert e['premium']==r['datasets'][e['dataset_id']]['premium']==fitted['premium']
    for role in e['roles']:
        if role in ('original','task_selected'): assert e['candidate_index']==0
        elif role=='reference': assert e['candidate_index']==1
    g=r['feature_groups'].get(e['feature_group'],{})
    equal(e['training_s'],sum(r['fits'][k]['total_s'] for k in e['refit_ids'])+g.get('draw_s',0.)+sum(g.get('splits_s',{}).get(k,0.) for k in ('train','val')))
    equal(e['deployment_s'],e['feature_generation_s']+e['evaluation_s'])
    assert e['save_s']>=0 and e['evaluation_s']>=0

# Independently aggregate NPZ-derived quantities; no report functions used.
def read_csv(name):
    with (OUT/name).open(newline='') as handle:
        return list(csv.DictReader(handle))
D=read_csv('dataset_summary.csv'); S=read_csv('summary.csv'); P=read_csv('paired_transfer.csv'); F=read_csv('per_run.csv')
assert len(D)==78 and len(S)==26 and len(P)==6 and len(F)==360
keys=('net_mse','mean_error','mean_turnover','mean_notional_turnover','mean_proportional_cost','mean_temporary_cost','mean_cost')
dataset_metrics={}
for row in D:
    key=(row['task'],row['condition'],row['role'],int(row['split_index']))
    es=[e for e in r['evaluations'].values() if (e['task'],e['condition'],e['split_index'])==(key[0],key[1],key[3]) and key[2] in e['roles']]
    assert len(es)==(1 if key[2] in ('simple','transformer') else 3)
    vals={k:np.mean([metrics_by_id[e['id']][k] for e in es]) for k in keys}
    vals['net_rmse']=np.sqrt(vals['net_mse'])
    for k in ('training_s','deployment_s','evaluation_s','feature_generation_s'):
        vals[k]=np.mean([e[k] for e in es])
    vals['final_compute_s']=vals['training_s']+vals['deployment_s']
    for k,v in vals.items(): equal(float(row[k]),v)
    dataset_metrics[key]=vals
for row in S:
    gs=[dataset_metrics[(row['task'],row['condition'],row['role'],i)] for i in range(3)]
    for k in (*keys,'training_s','deployment_s','evaluation_s','feature_generation_s','final_compute_s'):
        equal(float(row[k]),np.mean([g[k] for g in gs]))
    equal(float(row['net_rmse']),np.sqrt(np.mean([g['net_mse'] for g in gs])))
    equal(float(row['dataset_rmse_sd']),np.std([g['net_rmse'] for g in gs],ddof=1))
    equal(float(row['dataset_compute_sd']),np.std([g['final_compute_s'] for g in gs],ddof=1))
for row in P:
    ds=[dataset_metrics[(row['task'],row['condition'],'reference',i)]['net_rmse']-dataset_metrics[(row['task'],row['condition'],'task_selected',i)]['net_rmse'] for i in range(3)]
    equal(float(row['reference_minus_target_rmse']),np.mean(ds))
    equal(float(row['dataset_difference_sd']),np.std(ds,ddof=1))
    for i,v in enumerate(ds): equal(float(row[f'split_{i}_difference']),v)
assert {f['id'] for f in F}==set(r['fits'])
for row in F:
    fit=r['fits'][row['id']]
    assert row['status']=='completed' and row['condition']==fit['condition']
    equal(float(row['premium']),fit['premium'])
    equal(float(row['total_s']),fit['total_s'])
    if row['evaluation_id']:
        for k,v in metrics_by_id[row['evaluation_id']].items(): equal(float(row['test_'+k]),v)
assert len({g['id'] for g in r['feature_groups'].values()})==len(r['feature_groups'])
assert r['process_peak_rss_bytes']>0
assert all(g['process_peak_rss_bytes']>=g['rss_after_bytes'] and all(n>=0 for n in g['feature_bytes'].values()) for g in r['feature_groups'].values())
output={'status':'passed','audit':'Independent direct saved-NPZ to CSV verification; no fit or report functions invoked',
 'fits':len(F),'unique_evaluations':len(metrics_by_id),'role_comparisons':186,
 'regenerated_dataset_partitions':len(r['datasets']),'regenerated_path_sets':3*len(r['datasets']),
 'summary_rows':len(S),'dataset_summary_rows':len(D),'paired_rows':len(P),
 'checks':['source and reference hashes','all dataset spot hashes and manually discounted payoffs','training-only fixed premiums','checkpoint and shortlist selection','all saved net-error identities and cash-flow component identities','zero costs and nonnegative charges/turnover','unique configuration evaluations and role aliases','all direct NPZ-derived summary CSV metrics','three-dataset sample SD and paired differences','per-run CSV fit and final metrics','training/deployment timing identities','unique cache groups and memory records'],
 'limitations':['Cash-flow positions/trade arrays are not stored; detailed execution and gradient identities are covered by passing component tests.','Memory uses cumulative process peak and feature-array bytes, not isolated model allocations.','Three fixed reservoir draws are shared across datasets; dataset SD is conditional descriptive variation, not a confidence interval.'],
 'optimizer_outcomes':dict(Counter(f['optimizer_status'] for f in r['fits'].values()))}
(OUT/'independent_audit.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
