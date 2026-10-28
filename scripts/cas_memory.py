"""Bounded frictionless memory comparison: declare, run/resume, report only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

# Set before importing NumPy; all reported study computations use two threads.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cas_frictionless import write_json, timed
from cas_transfer_report import _write_csv
from rsmc.control import RidgeMartingaleControl, control_features
from rsmc.cost_hedging import hedge_readout_features
from rsmc.features import PathFeatures
from rsmc.hedging import FrozenTanhFeatures, discounted_traded_increments, evaluate_hedge, fit_fixed_premium_hedge
from rsmc.reservoir import RandomizedSignatureReservoir
from rsmc.transfer_tasks import make_transfer_dataset

OUT = ROOT / "results/cas_memory"
METHODS = ["affine", "static", "reservoir"]
TASKS = ["black_scholes", "heston", "basket"]
CANDIDATES = [dict(dim=64, weight_scale=s, bias_scale=1., lam=l)
              for s in (.5, 1.) for l in (0., .1, 10.)]
PLAN = {
    "tasks": TASKS, "methods": METHODS, "n_steps": 50, "strike": 100.,
    "sizes": {"train": 4000, "val": 2000, "test": 8000},
    "candidates": CANDIDATES, "affine_lambdas": [0., .1, 10.],
    "datasets": [dict(id=f"{task}_{b}", task=task, partition=b,
                      seed=2026102800+100*a+b,
                      tuning_seeds={m:[2026103800+1000*a+100*b+10*i+j for j in (1,2,3)]
                                    for i,m in enumerate(METHODS[1:])},
                      final_seeds={m:[2026113800+1000*a+100*b+10*i+j for j in (1,2,3)]
                                   for i,m in enumerate(METHODS[1:])})
                 for a,task in enumerate(TASKS) for b in (1,2,3)],
    "selection": "Separately for each of nine datasets and each method: minimum unpenalized validation MSE, mean over all three tuning draws for nonlinear methods; candidate order breaks exact ties. All 27 selections freeze before test generation. No incomplete grid, extra starts, replacement seeds or test selection.",
    "final": "Same partition training paths, three fresh draws per selected nonlinear method and one explicit affine refit; no train+validation refit. Never select a draw or average holdings. Test child stream generated only after all selections freeze.",
    "information": "Existing price-only time/log current constituent prices/log running payoff average. Same [1,t,current non-time summaries] passthrough; no fitted input normalization.",
    "ridge": "Training-only gain-column RMS q (values below1e-12 replaced by1); solve sum(H-p-Psi/q*gamma)^2+lambda*||gamma||^2; beta=gamma/q. All coefficients penalized. No centering; mean-loss penalty lambda/4000.",
    "accounting": "Premium float32(training discounted payoff mean), frozen across methods and splits. Gains=sum theta_k*(discounted S_{k+1}-discounted S_k), all traded constituents. Frictionless, no Brownian gains or test recentering.",
    "budget": {"nonlinear_selection":324,"affine_selection":27,"nonlinear_final":54,"affine_final":9,"total":414},
    "compute": "CPU float64, two BLAS/OpenMP threads. Reuse frozen features/gain rows across three lambdas. Record all generation (including resume regeneration), solves, validation, test and artifact times; complete invocation wall separately.",
    "protocol_revision": "User instruction supersedes original 180-fit design: selection within each of three partitions per task, rather than a fourth selection partition. Width and six tuples unchanged; fresh final draw namespace is tuning+10000.",
}


def configure_windows():
    """The single declared extension; historical constants remain the default."""
    global OUT, METHODS, PLAN
    from dataclasses import asdict
    from rsmc.transfer_tasks import BlackScholesModel, HestonModel, MultiAssetHestonModel
    from rsmc.multi_asset_heston import default_multi_asset_heston_correlation
    OUT = ROOT / "results/cas_memory/windows"
    METHODS = ["static", "window5", "window20", "reservoir"]
    PLAN = {**PLAN,
        "methods": METHODS, "windows": {"static": 0, "window5": 5, "window20": 20},
        "path_batch_size": 2000,
        "datasets": [dict(id=f"{task}_{b}", task=task, partition=b,
            seed=2026200000+100*a+b,
            tuning_seeds={m:[2026210000+1000*a+100*b+10*i+j for j in (1,2,3)]
                          for i,m in enumerate(METHODS)},
            final_seeds={m:[2026220000+1000*a+100*b+10*i+j for j in (1,2,3)]
                         for i,m in enumerate(METHODS)})
            for a,task in enumerate(TASKS) for b in (1,2,3)],
        "selection": "Separately for each of nine datasets and four fixed representations: minimum unpenalized validation MSE averaged over all three tuning draws; candidate order breaks exact ties. Freeze all 36 settings before test generation. Windows are never selected against one another. No extra starts, replacement seeds or test selection.",
        "final": "Refit selected settings on the same training paths with three fresh draws per representation. No train+validation refit, draw selection or averaged holdings. Shared fresh test child streams open only after all selections freeze.",
        "window_inputs": "Concatenate x_k,x_max(k-1,0),...,x_max(k-w,0), including all existing time/price/running-average channels; w=5/20 means 6/21 vectors. Expand only the nonlinear input; the affine passthrough remains current-summary only.",
        "random_scaling": "Static/window A has shape (64,d*(w+1)), Gaussian SD s/sqrt(d*(w+1)); b has SD1. Reservoir channel matrices have shape (64,64), Gaussian SD s/sqrt(64), biases SD1. Draw matrices before biases; reuse base draws across scales and penalties. Recurrence unchanged.",
        "budget": {"nonlinear_selection":648,"affine_selection":0,"nonlinear_final":108,"affine_final":0,"total":756},
        "aggregation": "Average squared errors over draws within each partition, then equally over three partitions. RMSE is sqrt(pooled MSE). Report partition sample SD and within-partition draw variation separately, no confidence intervals. Save all six paired differences on shared paths; figure shows each tanh representation minus reservoir.",
        "tail_diagnostics": "Retain every path. Report top1/top5 shared-path SSE shares per method and signed/absolute contributions of the five largest absolute pathwise paired differences. These are descriptive diagnostics, never selection criteria.",
        "compute": "CPU float64, two BLAS/OpenMP threads, 2000-path feature batches for every method. Reuse gain rows across three penalties. Include lag construction in feature timing; save draw, feature, gain, fit, validation, evaluation, regeneration and complete process timings. Retried interrupted attempts remain visible.",
        "protocol_revision": "Bounded history-window extension: four fixed representations, no affine, Transformer, depth-two or cost-aware reruns. Historical artifacts are untouched.",
    }
    PLAN.pop("affine_lambdas")
    models={name:asdict(cls()) for name,cls in zip(TASKS,[BlackScholesModel,HestonModel,MultiAssetHestonModel])}
    models["basket"]["correlation"]=default_multi_asset_heston_correlation().tolist()
    PLAN["models"]=json.loads(json.dumps(models))
    PLAN["payoff"]="Discounted strike-100 arithmetic Asian call; average dates1..50, excluding inception. Basket weights1/3 each. Summary running averages include inception; no latent variance or innovations enter policies."


def feature_map(method, candidate, seed, channels):
    if method == "affine":
        return None
    cls = RandomizedSignatureReservoir if method == "reservoir" else FrozenTanhFeatures
    options = {"window": PLAN.get("windows", {}).get(method, 0)} if cls is FrozenTanhFeatures else {}
    return cls.draw(channels, candidate["dim"], np.random.default_rng(seed),
                    weight_scale=candidate["weight_scale"], bias_scale=candidate["bias_scale"], **options)


def gain_rows(data, times, rate, model):
    features = data["features"]
    n = len(features.values)
    chunks, feature_s, gain_s = [], 0., 0.
    for start in range(0, n, PLAN.get("path_batch_size", n)):
        stop = min(start + PLAN.get("path_batch_size", n), n)
        batch = PathFeatures(features.values[start:stop], features.names)
        phi, seconds = timed(lambda: hedge_readout_features(batch, times, model))
        feature_s += seconds
        psi, seconds = timed(lambda: control_features(phi, discounted_traded_increments(data["spot"][start:stop], times, rate)))
        gain_s += seconds
        chunks.append(psi)
    rows, seconds = timed(lambda: np.concatenate(chunks))
    return rows, feature_s, gain_s + seconds


def render(results):
    """Aggregate saved residuals; produce one two-panel figure and compact tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_rows, paired, tails = [], [], []
    for ds in PLAN["datasets"]:
        archive = np.load(OUT/f"test_{ds['id']}.npz")
        row_mse = {}
        for method in METHODS:
            records = [r for r in results["fits"].values() if r["dataset"] == ds["id"] and r["phase"] == "final" and r["method"] == method]
            errors = np.stack([archive[r["id"]] for r in records])
            mse = float(np.mean(errors**2))
            groups = [g for g in results["groups"] if g["dataset"]==ds["id"] and g["method"]==method]
            selection = [r for r in results["fits"].values() if r["dataset"]==ds["id"] and r["method"]==method and r["phase"]=="tuning"]
            selection_s = sum(g["draw_s"]+sum(g["feature_s"].values())+sum(g["gain_s"].values()) for g in groups if g["phase"]=="tuning") + sum(
                r["fit_s"]+r["validation_s"]+sum(a.get("fit_s",0.)+a.get("validation_s",0.) for a in r.get("interrupted_attempts",[])) for r in selection)
            # Canonical completed-group costs exclude extra regeneration attempts.
            final = list({g["id"]:g for g in groups if g["phase"]=="final"}.values())
            dataset_rows.append(dict(task=ds["task"],dataset=ds["id"],method=method,
                mse=mse,rmse=float(np.sqrt(mse)),mean_error=float(errors.mean()),
                draw_mse_sd=float(np.std(np.mean(errors**2,axis=1),ddof=1)) if len(records)>1 else 0.,
                draw_mse_min=float(np.min(np.mean(errors**2,axis=1))),
                draw_mse_max=float(np.max(np.mean(errors**2,axis=1))),
                trained_parameters=records[0]["readout_parameters"],
                frozen_parameters=final[0].get("frozen_parameters"),
                selection_s=selection_s,
                final_feature_s=sum(g["feature_s"]["train"] for g in final),
                final_fit_s=sum(r["fit_s"] for r in records),
                final_draw_s=sum(g["draw_s"] for g in final),
                final_gain_s=sum(g["gain_s"]["train"] for g in final),
                final_validation_s=sum(r["validation_s"] for r in records),
                single_draw_training_s=(sum(g["draw_s"]+g["feature_s"]["train"]+g["gain_s"]["train"] for g in final)+sum(r["fit_s"] for r in records))/len(records),
                test_feature_s=sum(g["feature_s"]["test"] for g in final),
                test_gain_s=sum(g["gain_s"]["test"] for g in final),
                evaluation_s=sum(r["evaluation_s"] for r in records),
                # Single-draw deployment, not an ensemble or total across three draws.
                deployment_s=sum(g["feature_s"]["test"]+g["gain_s"]["test"] for g in final)/len(records)+sum(r["evaluation_s"] for r in records)/len(records)))
            row_mse[method] = np.mean(errors**2,axis=0)
            order=np.argsort(row_mse[method])[::-1]
            total=float(row_mse[method].sum())
            tails.append(dict(dataset=ds["id"],method=method,top_path=int(order[0]),
                top1_sse_share=float(row_mse[method][order[0]]/total) if total else 0.,
                top5_sse_share=float(row_mse[method][order[:5]].sum()/total) if total else 0.,
                top5_paths=order[:5].tolist()))
        differences=dict(task=ds["task"],dataset=ds["id"])
        for left,right in combinations(METHODS,2):
            delta=row_mse[left]-row_mse[right]
            name=f"{left}_minus_{right}"
            differences[name]=float(delta.mean())
            order=np.argsort(np.abs(delta))[::-1][:5]
            absolute=float(np.abs(delta).sum())
            tails.append(dict(dataset=ds["id"],comparison=name,mean_difference=float(delta.mean()),
                top5_paths=order.tolist(),top5_signed_contribution=float(delta[order].sum()/len(delta)),
                top5_absolute_share=float(np.abs(delta[order]).sum()/absolute) if absolute else 0.,
                top5_path_differences=delta[order].tolist()))
        paired.append(differences)
    summary=[]
    for task in TASKS:
        for method in METHODS:
            rows=[r for r in dataset_rows if r["task"]==task and r["method"]==method]
            summary.append(dict(task=task,method=method,mse=float(np.mean([r["mse"] for r in rows])),
                rmse=float(np.sqrt(np.mean([r["mse"] for r in rows]))),
                dataset_mse_sd=float(np.std([r["mse"] for r in rows],ddof=1)),
                dataset_rmse_sd=float(np.std([r["rmse"] for r in rows],ddof=1)),
                mean_within_dataset_draw_mse_sd=float(np.mean([r["draw_mse_sd"] for r in rows])),
                trained_parameters=rows[0]["trained_parameters"],frozen_parameters=rows[0]["frozen_parameters"],
                mean_error=float(np.mean([r["mean_error"] for r in rows])),
                selection_s=sum(r["selection_s"] for r in rows),
                deployment_s=float(np.mean([r["deployment_s"] for r in rows]))))
    _write_csv(OUT/"dataset_summary.csv",dataset_rows)
    _write_csv(OUT/"summary.csv",summary)
    _write_csv(OUT/"paired.csv",paired)
    results["summary"] = summary
    results["tail_diagnostics"] = tails
    results["timing_totals"] = {}
    for method in METHODS:
        groups=[g for g in results["groups"] if g["method"]==method]
        fits=[r for r in results["fits"].values() if r["method"]==method]
        results["timing_totals"][method]=dict(
            draw_s=sum(g["draw_s"] for g in groups),
            feature_s=sum(sum(g["feature_s"].values()) for g in groups),
            gain_s=sum(sum(g["gain_s"].values()) for g in groups),
            fit_s=sum(r["fit_s"]+sum(a.get("fit_s",0.) for a in r.get("interrupted_attempts",[])) for r in fits),
            validation_s=sum(r["validation_s"]+sum(a.get("validation_s",0.) for a in r.get("interrupted_attempts",[])) for r in fits),
            evaluation_s=sum(sum(r.get("evaluation_attempts_s",[r.get("evaluation_s",0.)])) for r in fits),
            interrupted_fit_attempts=sum(len(r.get("interrupted_attempts",[])) for r in fits))
    if "windows" in PLAN:
        fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout="constrained")
        for ax,task,title in zip(axes,TASKS,["Black–Scholes Asian","Heston Asian","Heston basket Asian"]):
            means=[]
            for i,method in enumerate(METHODS[:-1]):
                values=np.array([r[f"{method}_minus_reservoir"] for r in paired if r["task"]==task])
                means.append(float(values.mean()))
                ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),fmt="o",color="black",capsize=4,markersize=4)
                ax.scatter(i+np.array([-.09,0,.09]),values,color="tab:blue",s=23,zorder=3)
            ax.axhline(0,color="gray",ls="--",lw=1)
            ax.set(xticks=range(3),xticklabels=["Current","Window 5","Window 20"],title=title,
                ylabel="Tanh MSE − reservoir MSE (currency²)")
            ax.text(.5,-.17,"Means: "+", ".join(f"{v:+.3g}" for v in means),ha="center",transform=ax.transAxes,fontsize=9)
        fig.suptitle("Paired test error differences; three partitions per task")
        fig.text(.01,-.05,"Dots: partitions after averaging draws. Black: mean ± partition SD, not a confidence interval. Positive favors reservoir; panel scales differ.",fontsize=9)
        fig.savefig(OUT/"comparison.png",dpi=170,bbox_inches="tight")
        plt.close(fig)
        return
    fig,axes=plt.subplots(1,2,figsize=(12.4,5.2),layout="constrained")
    offsets=(-.07,0,.07)
    for i,task in enumerate(TASKS):
        values=[r["static_minus_reservoir"] for r in paired if r["task"]==task]
        axes[0].scatter([i+o for o in offsets],values,color="#176b87",s=35,zorder=3)
        axes[0].plot([i-.18,i+.18],[np.mean(values)]*2,color="black",lw=2)
    axes[0].set_yscale("symlog",linthresh=.2)
    axes[0].set_yticks([-.3,-.1,0,.1,.5,1,10,30],labels=["−0.3","−0.1","0","0.1","0.5","1","10","30"])
    axes[0].axhline(0,color="gray",ls="--",lw=1)
    labels=["Black–Scholes","Heston","Heston basket"]
    axes[0].set(xticks=range(3),xticklabels=labels,ylabel="Static MSE − reservoir MSE (currency units²)",title="Error difference on shared test paths\nSigned log scale; linear between −0.2 and 0.2")
    for m,method in enumerate(METHODS):
        values=[]
        for i,task in enumerate(TASKS):
            rows=[r for r in dataset_rows if r["task"]==task and r["method"]==method]
            x=i+(m-1)*.24
            axes[1].bar(x,np.mean([r["selection_s"] for r in rows]),width=.21,color=["#888888","#d88a22","#176b87"][m],label=method.capitalize() if i==0 else None,alpha=.8)
            axes[1].scatter([x+o*.5 for o in offsets],[r["selection_s"] for r in rows],color="black",s=12,zorder=3)
    axes[1].set(xticks=range(3),xticklabels=labels,yscale="log",ylabel="Full selection cost per partition (seconds, log scale)",title="Measured features + fitting + validation")
    axes[1].legend(frameon=False,loc="upper left")
    fig.suptitle("Three partitions per task; feature draws averaged within each partition",fontsize=12)
    fig.text(.01,-.05,"Left: dots = paired partition differences; black lines = means (positive favors reservoir). Right: bars = means; dots = partitions. No confidence intervals.",fontsize=9)
    fig.savefig(OUT/"comparison.png",dpi=170,bbox_inches="tight")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare",action="store_true")
    parser.add_argument("--report-only",action="store_true")
    parser.add_argument("--windows",action="store_true",help="Run the fixed 756-fit history-window extension.")
    args=parser.parse_args()
    if args.windows: configure_windows()
    OUT.mkdir(parents=True,exist_ok=True)
    plan_path=OUT/"plan.json"
    source_paths=[Path(__file__),plan_path,*sorted((ROOT/"rsmc").glob("*.py"))]
    if args.windows: source_paths += [ROOT/"scripts/cas_frictionless.py",ROOT/"scripts/cas_transfer_report.py"]
    if args.declare:
        if plan_path.exists() and json.loads(plan_path.read_text())!=PLAN:
            raise RuntimeError("Existing declaration differs; refusing overwrite.")
        if not plan_path.exists(): write_json(plan_path,PLAN)
        if args.windows:
            sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
            declaration=OUT/"declaration.json"
            if declaration.exists():
                if json.loads(declaration.read_text())["source_sha256"]!=sources:
                    raise RuntimeError("Sources differ from frozen declaration.")
            else:
                seeds=[d["seed"] for d in PLAN["datasets"]]
                seeds += [s for d in PLAN["datasets"] for phase in ("tuning_seeds","final_seeds") for draws in d[phase].values() for s in draws]
                assert len(seeds)==len(set(seeds))==225
                def integers(value):
                    if type(value) is int: yield value
                    elif isinstance(value,dict):
                        for v in value.values(): yield from integers(v)
                    elif isinstance(value,list):
                        for v in value: yield from integers(v)
                checked=0
                for old in (ROOT/"results").rglob("*.json"):
                    if OUT in old.parents: continue
                    if set(seeds).intersection(integers(json.loads(old.read_text()))):
                        raise RuntimeError(f"Seed collision with {old}")
                    checked+=1
                (OUT/"provenance").mkdir(exist_ok=True)
                (OUT/"provenance/execution_source.py").write_bytes(Path(__file__).read_bytes())
                write_json(declaration,dict(declared_utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=sources,unique_seeds=len(seeds),seed_lists="plan.json datasets",
                    seed_check=f"No proposed seed occurs as an integer value in {checked} prior result JSON files; no paths generated.",
                    fit_budget=PLAN["budget"]))
        print(f"Declared {PLAN['budget']['total']} fits; {len(PLAN['datasets'])*len(METHODS)} separately selected settings.")
        return
    if json.loads(plan_path.read_text())!=PLAN: raise RuntimeError("Code differs from frozen plan.")
    path=OUT/"results.json"
    results=json.loads(path.read_text()) if path.exists() else dict(status="running",fits={},groups=[],datasets={},selection={},failures=[],invocations=[])
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    if args.windows and json.loads((OUT/"declaration.json").read_text())["source_sha256"]!=sources:
        raise RuntimeError("Sources differ from the pre-execution declaration.")
    if results.get("source_sha256",sources)!=sources:
        # A completed study may rerender after a plotting-only revision. Preserve
        # the exact executed runner and keep every numerical-module hash fixed.
        original=results["source_sha256"]
        archive=OUT/"provenance/execution_source.py"
        self_key=str(Path(__file__).relative_to(ROOT))
        if (not args.report_only or not archive.exists()
            or hashlib.sha256(archive.read_bytes()).hexdigest()!=original[self_key]
            or any(sources[k]!=v for k,v in original.items() if k!=self_key)):
            raise RuntimeError("Execution sources changed; refusing mixed results.")
        results["reporting_source_sha256"]=sources[self_key]
    else:
        results["source_sha256"]=sources
    if args.report_only:
        if results["status"]!="completed": raise RuntimeError("Cannot report incomplete study.")
        render(results)
        write_json(path,results)
        return
    started=time.perf_counter()
    invocation=dict(started_utc=datetime.now(timezone.utc).isoformat())
    results["invocations"].append(invocation)
    results["environment"]=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown'),
        threads=2,precision="float64",numpy_config=str(np.__config__.show(mode="dicts")))
    def save(): write_json(path,results)
    def dataset(ds,test):
        sizes=PLAN["sizes"] if test else {k:v for k,v in PLAN["sizes"].items() if k!="test"}
        if test:
            assert sum(len(v) for v in results["selection"].values())==len(PLAN["datasets"])*len(METHODS)
            assert results["selection_sha256"]==hashlib.sha256(json.dumps(results["selection"],sort_keys=True).encode()).hexdigest()
        generation=dict(dataset=ds["id"],test_included=test,started_utc=datetime.now(timezone.utc).isoformat(),paths=sum(sizes.values()))
        invocation.setdefault("data_generation",[]).append(generation)
        save()
        (data,times,rate,premium,meta),seconds=timed(lambda:make_transfer_dataset(ds["task"],sizes,50,ds["seed"],100.))
        if isinstance(meta["model"].get("correlation"),np.ndarray): meta["model"]["correlation"]=meta["model"]["correlation"].tolist()
        previous=results["datasets"].get(ds["id"])
        if previous:
            assert premium==previous["premium"]
            for split,digest in previous["dataset_spot_sha256"].items():
                assert digest==meta["dataset_spot_sha256"][split]
        results["datasets"][ds["id"]]=meta
        generation.update(seconds=seconds,completed_utc=datetime.now(timezone.utc).isoformat())
        save()
        return data,times,rate,premium
    def group(ds,phase,method,candidate,seed,data,times,rate):
        channels=data["train"]["features"].n_channels
        model,draw_s=timed(lambda:feature_map(method,candidate,seed,channels))
        rows={}
        gid=f"{ds['id']}_{phase}_{method}_s{candidate.get('weight_scale')}_d{seed}"
        g=dict(id=gid,regeneration=any(g["id"]==gid for g in results["groups"]),dataset=ds["id"],phase=phase,method=method,seed=seed,weight_scale=candidate.get("weight_scale"),draw_s=draw_s,feature_s={},gain_s={},
            frozen_parameters=int(model.weights.size+model.biases.size) if model is not None else 0,
            started_utc=datetime.now(timezone.utc).isoformat(),status="running")
        results["groups"].append(g)
        save()
        for split in data:
            rows[split],g["feature_s"][split],g["gain_s"][split]=gain_rows(data[split],times,rate,model)
            save()
        g.update(status="completed",completed_utc=datetime.now(timezone.utc).isoformat())
        save()
        return rows
    def fit(key,ds,phase,method,index,seed,candidate,rows,data,premium):
        previous=results["fits"].get(key)
        if previous and previous["status"]=="completed": return previous
        if previous and previous["status"]=="failed": raise RuntimeError(f"Failed fit retained; no automatic replacement: {key}")
        interrupted=[] if previous is None else previous.get("interrupted_attempts",[])+[{k:v for k,v in previous.items() if k!="interrupted_attempts"}]
        record=dict(id=key,dataset=ds["id"],phase=phase,method=method,candidate_index=index,seed=seed,candidate=candidate,status="running",
            started_utc=datetime.now(timezone.utc).isoformat(),interrupted_attempts=interrupted)
        results["fits"][key]=record
        save()
        try:
            fitted,record["fit_s"]=timed(lambda:fit_fixed_premium_hedge(rows["train"],data["train"]["payoff"],premium,candidate["lam"]))
            val,record["validation_s"]=timed(lambda:evaluate_hedge(fitted,rows["val"],data["val"]["payoff"]))
            if not np.isfinite([val.error_mse,*fitted.beta]).all(): raise ValueError("Nonfinite fitted result")
            record.update(status="completed",validation_mse=val.error_mse,readout_parameters=len(fitted.beta),premium=premium,completed_utc=datetime.now(timezone.utc).isoformat())
            if phase=="final":
                scale=np.sqrt(np.mean(rows["train"]**2,axis=0))
                scale=np.where(scale<1e-12,1.,scale)
                record.update(beta=fitted.beta.tolist(),train_gain_rms=scale.tolist())
        except Exception:
            record.update(status="failed",traceback=traceback.format_exc())
            results["failures"].append(key)
        save()
        return record
    try:
        # Generate train/validation only; no test child streams are opened here.
        for ds in PLAN["datasets"]:
            if ds["id"] in results["selection"]: continue
            data,times,rate,premium=dataset(ds,False)
            choices={}
            for method in METHODS:
                candidates=[dict(lam=l) for l in PLAN["affine_lambdas"]] if method=="affine" else CANDIDATES
                for s in ([None] if method=="affine" else [.5,1.]):
                    indices=[i for i,c in enumerate(candidates) if c.get("weight_scale")==s]
                    for seed in ([0] if method=="affine" else ds["tuning_seeds"][method]):
                        keys=[f"{ds['id']}_tuning_{method}_c{i}_d{seed}" for i in indices]
                        if all(results["fits"].get(k,{}).get("status")=="completed" for k in keys): continue
                        rows=group(ds,"tuning",method,candidates[indices[0]],seed,data,times,rate)
                        for i,key in zip(indices,keys): fit(key,ds,"tuning",method,i,seed,candidates[i],rows,data,premium)
                scores=[]
                for i,candidate in enumerate(candidates):
                    records=[r for r in results["fits"].values() if r["dataset"]==ds["id"] and r["phase"]=="tuning" and r["method"]==method and r["candidate_index"]==i]
                    if len(records)!=(1 if method=="affine" else 3) or any(r["status"]!="completed" for r in records): raise RuntimeError("Incomplete selection grid; inspect failures.")
                    scores.append(float(np.mean([r["validation_mse"] for r in records])))
                index=min(range(len(scores)),key=lambda i:(scores[i],i))
                choices[method]=dict(candidate_index=index,candidate=candidates[index],validation_mse=scores[index],candidate_scores=scores)
            results["selection"][ds["id"]]=choices
            invocation.setdefault("selection_events",[]).append(dict(dataset=ds["id"],frozen_utc=datetime.now(timezone.utc).isoformat()))
            save()
            print("Selection frozen",ds["id"],{m:v["candidate_index"] for m,v in choices.items()},flush=True)
            del rows,data
        assert len(results["selection"])==9
        frozen=hashlib.sha256(json.dumps(results["selection"],sort_keys=True).encode()).hexdigest()
        assert results.get("selection_sha256",frozen)==frozen
        results["selection_sha256"]=frozen
        results.setdefault("all_selections_frozen_utc",datetime.now(timezone.utc).isoformat())
        save()
        # Only now open test child streams. All final draws share these paths.
        for ds in PLAN["datasets"]:
            array_path=OUT/f"test_{ds['id']}.npz"
            if array_path.exists(): continue
            data,times,rate,premium=dataset(ds,True)
            arrays={"payoff":data["test"]["payoff"]}
            for method in METHODS:
                selected=results["selection"][ds["id"]][method]
                for seed in ([0] if method=="affine" else ds["final_seeds"][method]):
                    key=f"{ds['id']}_final_{method}_d{seed}"
                    rows=group(ds,"final",method,selected["candidate"],seed,data,times,rate)
                    record=fit(key,ds,"final",method,selected["candidate_index"],seed,selected["candidate"],rows,data,premium)
                    if record["status"]!="completed": raise RuntimeError("Final fit incomplete; inspect failures.")
                    fitted=RidgeMartingaleControl(np.array(record["beta"]),premium,selected["candidate"]["lam"])
                    record.setdefault("first_test_evaluation_utc",datetime.now(timezone.utc).isoformat())
                    report,seconds=timed(lambda:evaluate_hedge(fitted,rows["test"],data["test"]["payoff"]))
                    record.setdefault("evaluation_s",seconds)
                    record.setdefault("evaluation_attempts_s",[]).append(seconds)
                    if not np.isfinite(report.errors).all(): raise ValueError("Nonfinite test errors")
                    record.update(test_mse=report.error_mse,test_rmse=report.rmse,mean_error=report.mean_error)
                    arrays[key]=report.errors
                    save()
            tick=time.perf_counter()
            temporary=array_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary,**arrays)
            temporary.replace(array_path)
            invocation.setdefault("array_save_s",[]).append(time.perf_counter()-tick)
            save()
            print("Test complete",ds["id"],flush=True)
            del rows,data,arrays
        assert len(results["fits"])==PLAN["budget"]["total"] and not results["failures"]
        assert all(r["status"]=="completed" for r in results["fits"].values())
        results["status"]="completed"
        render(results)
    except BaseException:
        results["status"]="incomplete"
        invocation["interruption"]=traceback.format_exc()
        raise
    finally:
        results["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        invocation["through_render_wall_s"]=time.perf_counter()-started
        save()
        # Include the final results write in the externally visible timing.
        elapsed=time.perf_counter()-started
        (OUT/"timing.json").write_text(json.dumps(dict(last_invocation_complete_wall_s=elapsed,
            measurement="Through final results save; includes figure/tables, excludes writing this tiny timing sidecar and any later editorial report."),indent=2)+"\n")
        print(f"{results['status']}: {len(results['fits'])}/{PLAN['budget']['total']} fit records; invocation {elapsed:.3f}s",flush=True)


if __name__=="__main__":
    main()
