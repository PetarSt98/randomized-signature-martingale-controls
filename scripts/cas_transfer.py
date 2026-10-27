"""Predeclared transfer of settings and three-partition reliability study.

Run --declare before execution. Completed fits are reused on resume; an
interrupted fit remains explicitly incomplete rather than receiving extra starts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch

from cas_frictionless import ROOT, PLAN as BASE, write_json, timed
from rsmc.transfer_tasks import make_transfer_dataset
from rsmc.reservoir import RandomizedSignatureReservoir, DepthTwoSignatureReservoir
from rsmc.hedging import fit_fixed_premium_hedge
from rsmc.cost_hedging import (CostSpec, execution_cashflows, cost_metrics,
    hedge_readout_features, gain_row_scale, fit_cost_readout, readout_positions)
from rsmc.cost_transformer import (CostTransformerFit, fit_cost_transformer,
    evaluate_cost_transformer)
from rsmc.transformer_hedging import TransformerHedgeConfig, _build_model

OUT = ROOT / "results/cas_transfer"
TASKS = ["black_scholes", "heston", "basket"]
CANDIDATES = [dict(architecture=a, dim=d, weight_scale=w, bias_scale=1., lam=l)
    for a, d, w, l in [("shallow",64,1.,0.), ("depth_two",32,1.,.1),
        ("shallow",32,1.,.1), ("shallow",16,.5,.1),
        ("depth_two",16,.5,.1), ("depth_two",64,1.,10.)]]
CONDITIONS = {"zero": CostSpec(0.,0.), "combined": CostSpec(.0005,.0005)}
PLAN = {
    "tasks": TASKS, "n_steps": 50, "strike": 100., "sizes": BASE["sizes"],
    "candidates": CANDIDATES, "original_index": 0,
    "original_source": "README quick start: shallow D64, scale1, bias1, lambda0; only its settings transfer from the historical pricing example into current fixed-premium traded-gain hedging",
    "reference_indices": {"zero": 1, "combined": 1},
    "reference_source": "results/cas_costs/results.json:reference_configuration, validation-selected overall reservoir; no reselection using historical test errors",
    "conditions": {k: asdict(v) for k,v in CONDITIONS.items()},
    "cost_units": "Illustrative uncalibrated kappa=eta=0.0005, reference quantity1 share per asset, per execution including entry/rebalance/liquidation. Costs sum exp(-rt)*S*(kappa*abs(q)+eta*q^2/q_ref) over dates and assets, no dt divisor or future-price impact.",
    "premium_rule": "float32-rounded discounted training payoff mean separately per dataset partition; shared by every method and both costs; no fitted premium or test recentering",
    "selection_seeds": dict(zip(TASKS, [821001,821002,821003])),
    "final_seeds": {task: [931000+100*i+j for j in (1,2,3)] for i,task in enumerate(TASKS)},
    "draw_seeds": {"tuning": [741001,741002,741003], "final": [751001,751002,751003]},
    "selection_rule": "Once per task on an independent selection partition: minimum mean unpenalized validation net MSE over3 reservoir draws after checkpoint/restart selection; candidate index breaks ties. Freeze all task selections before generating any final partition. Final validation selects only checkpoints/restarts, never tuples.",
    "final_rule": "Three new independent train/validation/test partitions per task and3 fresh reservoir draws; identical tuples share fits and evaluation records with explicit roles. Match paths and draw seeds across settings/cost conditions. Regenerate matrices for target input dimension, refit preprocessing and readouts on target training data.",
    "readout_optimizer": {"steps":120,"lr":.01,"grad_tol":1e-5},
    "restarts": ["fixed-premium zero-cost ridge", "ridge gamma plus N(0,0.01^2)"],
    "restart_seed_rule": "810000 + dataset_seed + draw_seed +100*candidate_index",
    "penalty": "net MSE + lambda/n_train * ||gamma||^2; gamma=train gain-row RMS * beta; Transformer .1/n_train * ||head||^2; AdamW weight_decay0",
    "simple_lambda": .1,
    "transformer": {**BASE["transformer"], "weight_decay":0.},
    "transformer_scope": "Existing one-asset Heston reference only; same architecture and20epoch limits,1zero start and2combined starts per final partition; no architecture search or BS/basket Transformer claims",
    "transformer_seeds": {"zero":[761001], "combined":[761001,761002]},
    "transformer_seed_rule": "listed seed +100*final_split_index",
    "budget": {"tuning":162,"final_reservoir_max":270,"simple":27,"transformer":9,"total_max":468},
    "dependence": "All roles/costs share paths within a dataset_id. Reservoir draws are shared across candidates and across final partitions; dataset summaries therefore describe conditional dataset variation for a fixed3-draw ensemble. Error bars are descriptive sample SD of3 dataset-level RMSEs, not confidence intervals or9 independent runs. Validation bars are SD across3 draws conditional on one selection partition.",
    "caching": "One dataset and one candidate/draw feature group in RAM; reuse train/val features across costs/restarts, discard after group. No persistent feature cache. Completed coefficients/checkpoints/results reused on resume; regenerated data/features and their overhead recorded separately. Test features generated only after all task selections freeze.",
    "memory": "Feature array nbytes plus boundary RSS from /proc/self/statm and process high-water ru_maxrss (Linux KiB converted to bytes); cumulative process peak, not isolated per-fit allocations. CPU only,2 threads.",
    "scope": "Existing implemented policy classes and bounded optimization; no global optimum, new inputs, architectures, simulators, filtering or reconstruction mathematics.",
}


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k:json_value(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [json_value(v) for v in value]
    return value


def memory():
    rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return {"rss_bytes":rss, "process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare",action="store_true")
    parser.add_argument("--report-only",action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    plan_file = OUT / "plan.json"
    if args.declare:
        if plan_file.exists() and json.loads(plan_file.read_text()) != PLAN:
            raise RuntimeError("Declaration already exists and differs; refusing alteration.")
        if not plan_file.exists():
            write_json(plan_file, PLAN)
        print("Declared6 tuples/task/condition; independent task tuning then3 final partitions x3 draws; <=468 fits.")
        return
    if json.loads(plan_file.read_text()) != PLAN:
        raise RuntimeError("Saved declaration differs from code.")
    path = OUT / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {
        "status":"running", "fits":{}, "datasets":{}, "feature_groups":{},
        "evaluations":{}, "selection":{}, "failures":[], "invocation_wall_s":[],
        "resume_overheads":[]}
    if args.report_only:
        from cas_transfer_report import render, audit_saved_runs
        audit_saved_runs(results,PLAN,OUT)
        render(results,PLAN,OUT)
        return
    started = time.perf_counter()
    sources = [Path(__file__), ROOT/"scripts/cas_frictionless.py", plan_file,
               *sorted((ROOT/"rsmc").glob("*.py"))]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if "source_sha256" in results and results["source_sha256"] != hashes:
        raise RuntimeError("Execution sources changed; refusing mixed results.")
    results["source_sha256"] = hashes
    results["versions"] = {"numpy":np.__version__,"torch":torch.__version__}
    results["torch_threads"] = torch.get_num_threads()
    historical = json.loads((ROOT/"results/cas_costs/results.json").read_text())
    for condition in CONDITIONS:
        assert historical["reference_configuration"][condition]["candidate"] == CANDIDATES[1]
    results["reference_results_sha256"] = hashlib.sha256((ROOT/"results/cas_costs/results.json").read_bytes()).hexdigest()

    def save():
        write_json(path,results)

    def fit(key,metadata,operation):
        if key in results["fits"]:
            record = results["fits"][key]
            if record["status"] != "completed":
                raise RuntimeError(f"Unfinished fit {key}; retained and counted, cannot silently restart beyond budget.")
            return record
        if len(results["fits"]) >= PLAN["budget"]["total_max"]:
            raise RuntimeError("Declared fit cap reached.")
        record = dict(id=key,status="running",**metadata)
        results["fits"][key] = record
        save()
        before = memory()
        tick = time.perf_counter()
        try:
            record.update(json_value(operation()))
            record["status"] = "completed"
        except Exception:
            record.update(status="failed",failure=traceback.format_exc())
            results["failures"].append({"id":key,"error":record["failure"]})
            results["status"] = "incomplete"
            raise
        finally:
            record.update(total_s=time.perf_counter()-tick,rss_before_bytes=before["rss_bytes"],
                          rss_after_bytes=memory()["rss_bytes"],process_peak_rss_bytes=memory()["process_peak_rss_bytes"])
            save()
            print(f"fit {len(results['fits']):3d}/<=468 {key}: {record['status']} ({record['total_s']:.2f}s)",flush=True)
        return record

    def dataset(task, phase, split_index):
        seed = PLAN["selection_seeds"][task] if phase == "tuning" else PLAN["final_seeds"][task][split_index]
        key = f"{task}_{phase}_{split_index}"
        bundle, elapsed = timed(lambda:make_transfer_dataset(task,PLAN["sizes"],PLAN["n_steps"],seed))
        data,times,rate,premium,metadata = bundle
        metadata.update(id=key,phase=phase,split_index=split_index,generation_s=elapsed,**memory())
        if key in results["datasets"]:
            old = results["datasets"][key]
            assert old["dataset_spot_sha256"] == metadata["dataset_spot_sha256"] and old["premium"] == premium
            results["resume_overheads"].append(dict(dataset_id=key,data_generation_s=elapsed))
        else:
            results["datasets"][key] = metadata
        save()
        return key,data,times,rate,premium

    def group(dataset_id,data,times,index,seed):
        meta = results["datasets"][dataset_id]
        key = f"{dataset_id}_c{index}_d{seed}"
        before = memory()
        tick = time.perf_counter()
        candidate = CANDIDATES[index] if index < 6 else None
        reservoir = None
        if candidate:
            cls = RandomizedSignatureReservoir if candidate["architecture"] == "shallow" else DepthTwoSignatureReservoir
            reservoir = cls.draw(data["train"]["features"].n_channels,candidate["dim"],np.random.default_rng(seed),
                weight_scale=candidate["weight_scale"],bias_scale=candidate["bias_scale"])
        record = dict(id=key,task=meta["task"],dataset_id=dataset_id,phase=meta["phase"],candidate_index=index,
            seed=seed,draw_s=time.perf_counter()-tick,splits_s={},feature_bytes={},rss_before_bytes=before["rss_bytes"])
        phi = {}
        for split in ("train","val"):
            phi[split],record["splits_s"][split] = timed(lambda:hedge_readout_features(data[split]["features"],times,reservoir))
            record["feature_bytes"][split] = phi[split].nbytes
        record.update(rss_after_bytes=memory()["rss_bytes"],process_peak_rss_bytes=memory()["process_peak_rss_bytes"])
        if key in results["feature_groups"]:
            results["resume_overheads"].append(dict(feature_group=key,regeneration_s=record["draw_s"]+sum(record["splits_s"].values())))
        else:
            results["feature_groups"][key] = record
        save()
        return key,reservoir,phi

    def metadata(dataset_id,index,seed,group_id,condition):
        d = results["datasets"][dataset_id]
        candidate = CANDIDATES[index] if index < 6 else {"architecture":"simple" if index==6 else "transformer","lam":.1}
        return dict(task=d["task"],dataset_id=dataset_id,dataset_seed=d["dataset_seed"],
            split_index=d["split_index"],phase=d["phase"],candidate_index=index,configuration=candidate,
            seed=seed,feature_group=group_id,condition=condition,method=candidate["architecture"],premium=d["premium"],lam=candidate["lam"])

    def ridge(dataset_id,data,times,rate,premium,index,seed,gid,phi):
        meta = metadata(dataset_id,index,seed,gid,"zero")
        def solve():
            (psi,scale),setup_s = timed(lambda:gain_row_scale(phi["train"],data["train"]["spot"],times,rate))
            fitted,fit_s = timed(lambda:fit_fixed_premium_hedge(psi,data["train"]["payoff"],premium,meta["lam"]))
            def validate():
                cash = execution_cashflows(readout_positions(phi["val"],fitted.beta,data["val"]["spot"]),data["val"]["spot"],times,rate,CONDITIONS["zero"])
                return cost_metrics(data["val"]["payoff"],premium,cash)
            scores,val_s = timed(validate)
            scores.pop("errors")
            return dict(beta=fitted.beta,scale=scale,validation_mse=scores["net_mse"],validation_metrics=scores,
                optimizer_status="exact_ridge",setup_s=setup_s,fit_s=fit_s,validation_s=val_s)
        return fit(f"{gid}_ridge",meta,solve)

    def positive(dataset_id,data,times,rate,premium,index,seed,gid,phi,initial):
        scale = np.asarray(initial["scale"])
        gamma = np.asarray(initial["beta"])*scale
        records = []
        for restart in range(2):
            start = gamma.copy()
            if restart:
                rseed = 810000+results["datasets"][dataset_id]["dataset_seed"]+seed+100*index
                start += np.random.default_rng(rseed).normal(0.,.01,start.shape)
            meta = dict(**metadata(dataset_id,index,seed,gid,"combined"),restart=restart)
            def optimize():
                fitted = fit_cost_readout(phi["train"],data["train"]["spot"],data["train"]["payoff"],
                    phi["val"],data["val"]["spot"],data["val"]["payoff"],times,rate,premium,
                    CONDITIONS["combined"],meta["lam"],scale,start,**PLAN["readout_optimizer"])
                fitted["optimizer_status"] = fitted.pop("status")
                return fitted
            records.append(fit(f"{gid}_combined_r{restart}",meta,optimize))
        return min(records,key=lambda r:(r["validation_mse"],r["restart"])),[initial["id"],*[r["id"] for r in records]]

    # Selection partitions are independent of all final partitions.
    for task in TASKS:
        did,data,times,rate,premium = dataset(task,"tuning",-1)
        scores = {condition:[] for condition in CONDITIONS}
        for index in range(6):
            draws = {condition:[] for condition in CONDITIONS}
            for seed in PLAN["draw_seeds"]["tuning"]:
                gid,reservoir,phi = group(did,data,times,index,seed)
                initial = ridge(did,data,times,rate,premium,index,seed,gid,phi)
                selected,_ = positive(did,data,times,rate,premium,index,seed,gid,phi,initial)
                draws["zero"].append(initial["validation_mse"])
                draws["combined"].append(selected["validation_mse"])
                del phi,reservoir
            for condition in CONDITIONS:
                scores[condition].append(dict(candidate_index=index,draw_scores=draws[condition],validation_mse=float(np.mean(draws[condition]))))
        selected = {condition:{**min(values,key=lambda r:(r["validation_mse"],r["candidate_index"])),"candidates":values} for condition,values in scores.items()}
        if task in results["selection"]:
            assert results["selection"][task] == selected
        results["selection"][task] = selected
        save()
        print(f"Selection frozen {task}: "+str({c:r['candidate_index'] for c,r in selected.items()}),flush=True)
        del data

    def evaluate(record,roles,refit_ids,data,times,rate,phi=None,feature_s=0.,evaluator=None):
        key = "test_"+record["id"]
        if key in results["evaluations"]:
            assert results["evaluations"][key]["roles"] == roles
            return
        condition = record["condition"]
        def apply():
            cash = execution_cashflows(readout_positions(phi,np.asarray(record["beta"]),data["test"]["spot"]),data["test"]["spot"],times,rate,CONDITIONS[condition])
            return cost_metrics(data["test"]["payoff"],record["premium"],cash),cash
        (metrics,cash),evaluation_s = timed(evaluator or apply)
        errors = metrics.pop("errors")
        tick = time.perf_counter()
        archive = {k:v for k,v in cash.items() if k != "trades"}
        array_file = f"{key}.npz"
        np.savez_compressed(OUT/array_file,errors=errors,payoff=data["test"]["payoff"],**archive)
        save_s = time.perf_counter()-tick
        g = results["feature_groups"].get(record["feature_group"],{})
        training_s = sum(results["fits"][fid]["total_s"] for fid in refit_ids)+g.get("draw_s",0.)+sum(g.get("splits_s",{}).get(s,0.) for s in ("train","val"))
        results["evaluations"][key] = {**{k:record[k] for k in ("task","dataset_id","dataset_seed","split_index","condition","method","candidate_index","seed","premium","feature_group")},
            "id":key,"fit_id":record["id"],"draw_seed":record["seed"],"roles":roles,"refit_ids":refit_ids,
            "metrics":metrics,"array_file":array_file,"evaluation_s":evaluation_s,"save_s":save_s,
            "feature_generation_s":feature_s,"deployment_s":feature_s+evaluation_s,"training_s":training_s}
        save()

    for task in TASKS:
        role_indices = {condition:{"original":0,"reference":1,"task_selected":results["selection"][task][condition]["candidate_index"]} for condition in CONDITIONS}
        indices = sorted({i for choices in role_indices.values() for i in choices.values()})
        for split_index in range(3):
            did,data,times,rate,premium = dataset(task,"final",split_index)
            for index in [*indices,6]:
                for seed in PLAN["draw_seeds"]["final"] if index<6 else [0]:
                    gid,reservoir,phi = group(did,data,times,index,seed)
                    initial = ridge(did,data,times,rate,premium,index,seed,gid,phi)
                    selected,refit_ids = (positive(did,data,times,rate,premium,index,seed,gid,phi,initial)
                        if index==6 or index in role_indices["combined"].values() else (None,[]))
                    test_phi,feature_s = timed(lambda:hedge_readout_features(data["test"]["features"],times,reservoir))
                    g = results["feature_groups"][gid]
                    g["splits_s"].setdefault("test",feature_s)
                    g["feature_bytes"].setdefault("test",test_phi.nbytes)
                    g["process_peak_rss_bytes"] = max(g["process_peak_rss_bytes"],memory()["process_peak_rss_bytes"])
                    for condition,record,ids in [("zero",initial,[initial["id"]]),("combined",selected,refit_ids)]:
                        roles = ["simple"] if index==6 else [role for role,i in role_indices[condition].items() if i==index]
                        if roles:
                            evaluate(record,roles,ids,data,times,rate,test_phi,feature_s)
                    del phi,test_phi,reservoir
            if task == "heston":
                for condition in CONDITIONS:
                    records = []
                    for raw_seed in PLAN["transformer_seeds"][condition]:
                        seed = raw_seed+100*split_index
                        config = TransformerHedgeConfig(**{**PLAN["transformer"],"seed":seed})
                        key = f"{did}_transformer_{condition}_{seed}"
                        def train():
                            fitted = fit_cost_transformer(data["train"]["features"],data["train"]["spot"],data["train"]["payoff"],
                                data["val"]["features"],data["val"]["spot"],data["val"]["payoff"],times,rate,premium,config,costs=CONDITIONS[condition],lam=.1)
                            torch.save(fitted.model.state_dict(),OUT/f"{key}.pt")
                            return dict(config=asdict(config),input_mean=fitted.input_mean,input_std=fitted.input_std,
                                validation_mse=fitted.validation_mse,best_epoch=fitted.best_epoch,history=fitted.history,
                                optimizer_status=fitted.status,gradient_inf=fitted.final_gradient_inf,timings=fitted.timings,n_parameters=fitted.n_parameters)
                        records.append(fit(key,metadata(did,7,seed,None,condition),train))
                    record = min(records,key=lambda r:(r["validation_mse"],r["seed"]))
                    def transformer_evaluate():
                        cfg = TransformerHedgeConfig(**record["config"])
                        model = _build_model(torch,torch.nn,3,1,cfg,premium)
                        model.load_state_dict(torch.load(OUT/f"{record['id']}.pt",weights_only=True))
                        fitted = CostTransformerFit(model,np.asarray(record["input_mean"]),np.asarray(record["input_std"]),cfg,premium,CONDITIONS[condition],
                            record["validation_mse"],record["best_epoch"],record["history"],record["optimizer_status"],record["gradient_inf"],record["timings"],record["n_parameters"])
                        output = evaluate_cost_transformer(fitted,data["test"]["features"],data["test"]["spot"],data["test"]["payoff"],times,rate)
                        return {**output["metrics"],"errors":output["errors"]},output["cashflows"]
                    evaluate(record,["transformer"],[r["id"] for r in records],data,times,rate,evaluator=transformer_evaluate)
            del data
    expected_roles = 3*2*3*3*3 + 3*2*3 + 2*3
    if sum(len(e["roles"]) for e in results["evaluations"].values()) != expected_roles or results["failures"]:
        raise RuntimeError("Declared comparisons incomplete; outputs retained.")
    results.update(status="completed",fit_count=len(results["fits"]),process_peak_rss_bytes=memory()["process_peak_rss_bytes"])
    results["invocation_wall_s"].append(time.perf_counter()-started)
    save()
    from cas_transfer_report import render, audit_saved_runs
    audit_saved_runs(results,PLAN,OUT)
    render(results,PLAN,OUT)
    print(f"Complete: {len(results['fits'])} fits, {len(results['evaluations'])} unique evaluations, {expected_roles} role comparisons.",flush=True)


if __name__ == "__main__":
    main()
