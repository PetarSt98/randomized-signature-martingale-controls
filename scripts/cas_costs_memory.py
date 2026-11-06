"""Cost-aware memory comparison: does reservoir history help once costs enter training?

Frictionless studies found no consistent reservoir advantage over a static tanh
map of the current summaries. With proportional execution costs the optimal
policy is no longer a function of the current summary alone: it depends on the
position already held, i.e. on history. This bounded study retrains every policy
class on the net-error objective *including* costs and asks whether history
representations (leaky integrator, reservoir, causal Transformer) gain on the
static map as costs grow.

Modes
-----
  --declare          save the frozen plan (refuses to overwrite a different one)
  --task NAME        run / resume one task (five tasks run as parallel processes)
  --report           aggregate all completed tasks into tables, figure, REPORT.md
  --smoke            tiny integration run into results/cas_costs_memory/smoke
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

THREADS = os.environ.get("RSMC_THREADS", "4")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, THREADS)
import numpy as np
import torch
torch.set_num_threads(int(THREADS))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cas_frictionless import timed
from cas_transfer_report import _write_csv


def write_json(path, value):
    """Atomic JSON write with retries: Windows indexers briefly lock fresh files."""
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(25):
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    path.write_text(text, encoding="utf-8")
from rsmc.cost_hedging import (
    CostSpec, cost_metrics, execution_cashflows, fit_cost_readout,
    gain_row_scale, hedge_readout_features, readout_positions,
)
from rsmc.cost_transformer import CostTransformerFit, evaluate_cost_transformer, fit_cost_transformer
from rsmc.features import PathFeatures
from rsmc.hedging import FrozenTanhFeatures, LeakyTanhFeatures, fit_fixed_premium_hedge
from rsmc.reservoir import RandomizedSignatureReservoir
from rsmc.transfer_tasks import make_transfer_dataset
from rsmc.transformer_hedging import TransformerHedgeConfig, _build_model

OUT = ROOT / "results" / "cas_costs_memory"
TASKS = ["black_scholes", "heston", "basket", "two_factor_heston", "volterra"]
FROZEN = ["affine", "static", "leaky", "reservoir"]
METHODS = FROZEN + ["transformer"]
CONDITIONS = {"zero": 0.0, "bp10": 0.001, "bp50": 0.005, "bp200": 0.02}
POSITIVE = [c for c in CONDITIONS if CONDITIONS[c] > 0]
CANDIDATES = [dict(dim=64, weight_scale=s, bias_scale=1.0, lam=l)
              for s in (0.5, 1.0) for l in (0.1, 10.0, 1000.0)]
AFFINE_CANDIDATES = [dict(lam=l) for l in (0.1, 10.0, 1000.0)]
OPTIMIZER = dict(optimizer="lbfgs", steps=200, grad_tol=1e-5, validate_every=20, smooth_abs=1e-3, dtype="float64")
LEAKY_DECAY = 0.8
PATH_BATCH = 2000
N_PARTITIONS = 3
SIZES = {"train": 4000, "val": 2000, "test": 8000}
TRANSFORMER = TransformerHedgeConfig(weight_decay=0.0)

PLAN = {
    "question": "Does a history representation (leaky integrator, reservoir, Transformer) beat the static current-summary tanh map once proportional costs enter the training objective?",
    "tasks": TASKS, "methods": METHODS, "n_steps": 50, "strike": 100.0, "sizes": SIZES,
    "conditions": {k: dict(kappa=v, eta=0.0, q_ref=1.0) for k, v in CONDITIONS.items()},
    "cost_units": "kappa is a proportional rate on discounted traded notional: 10, 50 and 200 basis points. Costs are charged at entry, every rebalance and terminal liquidation; no temporary impact, no price feedback.",
    "candidates": CANDIDATES, "affine_candidates": AFFINE_CANDIDATES,
    "leaky_decay": LEAKY_DECAY,
    "representations": {
        "affine": "[1, t, current non-time summaries]",
        "static": "affine passthrough + 64 frozen tanh(A x_k + b) of current summaries; no state",
        "leaky": "affine passthrough + leaky integrator q_k = 0.8 q_(k-1) + 0.2 tanh(A x_k + b); linear memory",
        "reservoir": "affine passthrough + 64-dim shallow randomized-signature reservoir state h_k",
        "transformer": "unchanged causal Transformer (d_model 64, 2 layers, window 16) with fixed-premium capital buffer",
        "none": "no trading; error = payoff - premium (reference, not fitted)",
    },
    "objective": "mean((H - p - gain + cost)^2) + lam/n_train * ||gamma||^2 in training-RMS-scaled gain coordinates; zero cost solved exactly by ridge; positive cost by full-batch L-BFGS (strong Wolfe) on the training objective with |q| smoothed to sqrt(q^2+eps^2)-eps, eps=0.001 shares. Validation, selection and test always use exact |q| costs.",
    "optimizer": OPTIMIZER,
    "optimizer_note": "Chosen after a timed trial on one Heston partition: 600 AdamW steps (lr 0.01-0.1) left the 200 bp validation error at 11.5-13.4 and still falling, while L-BFGS reached 9.7-10.0 within 200 iterations. Fits are chunked into 20 iterations between validations; the budget cap is 200 iterations and an iterate meeting the gradient tolerance or stalling the line search terminates early.",
    "starts": {"tuning": ["ridge"], "final": ["ridge"]},
    "starts_note": "ridge = that lambda's zero-cost ridge coefficients. A no-hedge start was timed and discarded: it stays far from the ridge-start validation error within the budget, so the 'no hedge' alternative is instead reported explicitly as the untraded reference and approached by the lambda=1000 candidate. Checkpoints are selected by unpenalized validation net MSE.",
    "premium": "float32(training discounted payoff mean) per partition, frozen across methods and cost conditions; never recentered",
    "selection": "Per task, partition, method and cost condition: argmin over candidates of validation net MSE of one tuning draw; ties by candidate order. Selections freeze before that partition's test paths are generated.",
    "protocol_revision": "Launched with two tuning draws; after 12 minutes the measured contended fit time (18-37 s) projected over four hours, so the run was stopped and re-declared with one tuning draw. Completed draw-1 fits were reused unchanged; draw-2 records remain in results.json as unused orphans. No test paths had been generated.",
    "final": "Three fresh draws per selected setting fitted on training paths only; all conditions share the same fresh draw seeds so cost comparisons are paired across conditions as well as methods.",
    "transformer": {**TRANSFORMER.__dict__, "head_penalty_lam": 0.1, "seeds": "one per partition, shared across conditions"},
    "transformer_partitions": [1],
    "transformer_note": "The Transformer is trained on partition 1 only (all four conditions), as in the earlier one-partition architecture and cost studies. Launched for all partitions; after measuring 11-minute contended runs it was restricted before any partition-2 Transformer fit began. Frozen-feature methods use all three partitions.",
    "risk_reporting": "net MSE, mean error (cost drag at fixed premium), error SD, 95% shortfall CVaR, share turnover and mean cost; no path trimming",
    "datasets": [dict(id=f"{task}_{b}", task=task, partition=b, seed=2026400000 + 100 * a + b,
                      tuning_seeds={m: [2026410000 + 1000 * a + 100 * b + 10 * i + j for j in (1,)]
                                    for i, m in enumerate(FROZEN[1:])},
                      final_seeds={m: [2026420000 + 1000 * a + 100 * b + 10 * i + j for j in (1, 2, 3)]
                                   for i, m in enumerate(FROZEN[1:])},
                      transformer_seed=2026430000 + 100 * a + b)
                 for a, task in enumerate(TASKS) for b in range(1, N_PARTITIONS + 1)],
    "nominal_budget_per_task": {
        "tuning_ridge": 3 * (3 + 3 * 6), "tuning_positive": 3 * (3 + 3 * 6) * 3,
        "final_ridge_min": 3 * (1 + 3 * 3), "final_ridge_max": 3 * (1 + 3 * 3) * 4,
        "final_positive": 3 * (1 + 3 * 3) * 3, "transformer": 1 * 4,
        "note": "Final ridge count depends on how many distinct lambdas the four conditions select per method/draw.",
    },
}


def feature_map(method, candidate, seed, channels):
    rng = np.random.default_rng(seed)
    if method == "affine":
        return None
    if method == "reservoir":
        return RandomizedSignatureReservoir.draw(channels, candidate["dim"], rng,
                                                 candidate["weight_scale"], candidate["bias_scale"])
    if method == "static":
        return FrozenTanhFeatures.draw(channels, candidate["dim"], rng,
                                       candidate["weight_scale"], candidate["bias_scale"])
    if method == "leaky":
        return LeakyTanhFeatures.draw(channels, candidate["dim"], rng,
                                      candidate["weight_scale"], candidate["bias_scale"],
                                      decay=PLAN["leaky_decay"])
    raise ValueError(method)


def readout_matrix(features, times, fmap):
    chunks = []
    for start in range(0, len(features.values), PATH_BATCH):
        batch = PathFeatures(features.values[start:start + PATH_BATCH], features.names)
        chunks.append(hedge_readout_features(batch, times, fmap))
    return np.concatenate(chunks)


def net_errors(theta, spot, times, rate, payoff, premium, kappa):
    cash = execution_cashflows(theta, spot, times, rate, CostSpec(kappa, 0.0))
    metrics = cost_metrics(payoff, premium, cash)
    return metrics


def run_task(task, out, plan, smoke=False):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    results = json.loads(path.read_text()) if path.exists() else dict(
        status="running", task=task, fits={}, groups={}, selection={}, evaluations={},
        datasets={}, failures=[], invocations=[])
    invocation = dict(started_utc=datetime.now(timezone.utc).isoformat(), threads=THREADS,
                      python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                      platform=platform.platform())
    results["invocations"].append(invocation)
    started = time.perf_counter()

    def save():
        write_json(path, results)

    def fit_record(key, operation, **meta):
        record = results["fits"].get(key)
        if record is not None and record["status"] == "completed":
            return record
        if record is not None and record["status"] == "failed":
            raise RuntimeError(f"Failed fit retained, not repeated: {key}")
        record = dict(id=key, status="running", started_utc=datetime.now(timezone.utc).isoformat(), **meta)
        results["fits"][key] = record
        save()
        tick = time.perf_counter()
        try:
            outcome = operation()
            if meta.get("phase") == "tuning":  # coefficients are kept for final fits only
                outcome = {k: v for k, v in outcome.items() if k not in ("beta", "scale", "history")}
            record.update(outcome)
            record["status"] = "completed"
        except Exception:
            record.update(status="failed", traceback=traceback.format_exc())
            results["failures"].append(key)
            save()
            raise
        record["total_s"] = time.perf_counter() - tick
        save()
        return record

    datasets = [d for d in plan["datasets"] if d["task"] == task]
    for ds in datasets:
        previous_evaluations = results["evaluations"].get(ds["id"], {})
        previous_archive = out / f"test_{ds['id']}.npz"
        if previous_evaluations.get("completed") and previous_archive.exists():
            saved = set(np.load(previous_archive).files)
            if all(k in saved for k in previous_evaluations if k != "completed"):
                continue
            print(f"[{task}] archive incomplete for {ds['id']}; recomputing test errors from saved fits", flush=True)
        selection_done = ds["id"] in results["selection"]
        sizes = plan["sizes"] if selection_done else {k: v for k, v in plan["sizes"].items() if k != "test"}
        (data, times, rate, premium, meta), gen_s = timed(
            lambda: make_transfer_dataset(task, sizes, plan["n_steps"], ds["seed"], plan["strike"]))
        if isinstance(meta["model"].get("correlation"), np.ndarray):
            meta["model"]["correlation"] = meta["model"]["correlation"].tolist()
        meta = json.loads(json.dumps(meta, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
        previous = results["datasets"].get(ds["id"])
        if previous:
            assert previous["premium"] == premium, "Premium changed on resume."
        results["datasets"][ds["id"]] = {**meta, "generation_s": gen_s, "test_included": selection_done}
        save()
        channels = data["train"]["features"].n_channels
        spot = {s: data[s]["spot"] for s in data}
        payoff = {s: data[s]["payoff"] for s in data}

        def group_matrices(method, candidate, seed, phase, splits):
            """Readout features for each split plus training gain rows and scales."""
            gid = f"{ds['id']}_{phase}_{method}_s{candidate.get('weight_scale')}_d{seed}"
            fmap, draw_s = timed(lambda: feature_map(method, candidate, seed, channels))
            phi, feature_s = {}, {}
            for split in splits:
                phi[split], feature_s[split] = timed(lambda: readout_matrix(data[split]["features"], times, fmap))
            (psi, scale), gain_s = timed(lambda: gain_row_scale(phi["train"], spot["train"], times, rate))
            results["groups"][gid] = dict(id=gid, dataset=ds["id"], phase=phase, method=method, seed=seed,
                weight_scale=candidate.get("weight_scale"), draw_s=draw_s, feature_s=feature_s, gain_s=gain_s,
                frozen_parameters=int(fmap.weights.size + fmap.biases.size) if fmap is not None else 0,
                readout_parameters=int(psi.shape[1]))
            save()
            return phi, psi, scale

        def ridge_fit(key, phase, method, index, seed, lam, phi, psi, scale):
            def solve():
                fitted, fit_s = timed(lambda: fit_fixed_premium_hedge(psi, payoff["train"], premium, lam))
                theta = readout_positions(phi["val"], fitted.beta, spot["val"])
                val = net_errors(theta, spot["val"], times, rate, payoff["val"], premium, 0.0)
                val.pop("errors")
                return dict(beta=fitted.beta.tolist(), gamma=(fitted.beta * scale).tolist(), scale=scale.tolist(),
                            validation_mse=val["net_mse"], validation_metrics=val, optimizer_status="exact_ridge", fit_s=fit_s)
            return fit_record(key, solve, dataset=ds["id"], phase=phase, method=method, condition="zero",
                              candidate_index=index, seed=seed, lam=lam)

        def positive_fits(key_base, phase, method, index, seed, lam, condition, phi, scale, ridge, starts):
            records = []
            for start in starts:
                init = np.asarray(ridge["gamma"]) if start == "ridge" else np.zeros(len(ridge["gamma"]))
                def optimize():
                    fitted = fit_cost_readout(phi["train"], spot["train"], payoff["train"], phi["val"], spot["val"],
                        payoff["val"], times, rate, premium, CostSpec(CONDITIONS[condition], 0.0), lam, scale, init,
                        **plan["optimizer"])
                    fitted["optimizer_status"] = fitted.pop("status")
                    fitted["beta"], fitted["gamma"] = fitted["beta"].tolist(), fitted["gamma"].tolist()
                    fitted["history"] = [fitted["history"][0], fitted["history"][-1]]  # keep records compact
                    return fitted
                records.append(fit_record(f"{key_base}_{condition}_{start}", optimize, dataset=ds["id"], phase=phase,
                    method=method, condition=condition, candidate_index=index, seed=seed, lam=lam, start=start))
            return min(records, key=lambda r: (r["validation_mse"], starts.index(r["start"])))

        if not selection_done:
            choices = {c: {} for c in CONDITIONS}
            for method in FROZEN:
                candidates = plan["affine_candidates"] if method == "affine" else plan["candidates"]
                seeds = [0] if method == "affine" else ds["tuning_seeds"][method]
                scores = {c: np.zeros(len(candidates)) for c in CONDITIONS}
                for seed in seeds:
                    for s in dict.fromkeys(c.get("weight_scale") for c in candidates):
                        indices = [i for i, c in enumerate(candidates) if c.get("weight_scale") == s]
                        keys = [f"{ds['id']}_tuning_{method}_c{i}_d{seed}" for i in indices]
                        needed = [k for k in keys if results["fits"].get(k, {}).get("status") != "completed"] + [
                            f"{k}_{c}_{st}" for k in keys for c in POSITIVE for st in plan["starts"]["tuning"]
                            if results["fits"].get(f"{k}_{c}_{st}", {}).get("status") != "completed"]
                        if needed:
                            phi, psi, scale = group_matrices(method, candidates[indices[0]], seed, "tuning", ("train", "val"))
                        for i, key in zip(indices, keys):
                            lam = candidates[i]["lam"]
                            if needed:
                                ridge = ridge_fit(key, "tuning", method, i, seed, lam, phi, psi, scale)
                                for condition in POSITIVE:
                                    positive_fits(key, "tuning", method, i, seed, lam, condition, phi, scale, ridge,
                                                  plan["starts"]["tuning"])
                            scores["zero"][i] += results["fits"][key]["validation_mse"] / len(seeds)
                            for condition in POSITIVE:
                                best = min(results["fits"][f"{key}_{condition}_{st}"]["validation_mse"]
                                           for st in plan["starts"]["tuning"])
                                scores[condition][i] += best / len(seeds)
                        if needed:
                            del phi, psi
                for condition in CONDITIONS:
                    index = int(min(range(len(candidates)), key=lambda i: (scores[condition][i], i)))
                    choices[condition][method] = dict(candidate_index=index, candidate=candidates[index],
                                                      validation_mse=float(scores[condition][index]),
                                                      candidate_scores=scores[condition].tolist())
            results["selection"][ds["id"]] = choices
            results["selection"][ds["id"] + "_frozen_utc"] = datetime.now(timezone.utc).isoformat()
            save()
            print(f"[{task}] selection frozen {ds['id']}", flush=True)
            # Regenerate with the test child stream; train/val are unchanged (checked above on resume).
            (data, times, rate, premium2, meta), gen_s = timed(
                lambda: make_transfer_dataset(task, plan["sizes"], plan["n_steps"], ds["seed"], plan["strike"]))
            assert premium2 == premium
            results["datasets"][ds["id"]].update(test_included=True, test_generation_s=gen_s,
                test_spot_sha256=meta["dataset_spot_sha256"]["test"])
            save()
            spot = {s: data[s]["spot"] for s in data}
            payoff = {s: data[s]["payoff"] for s in data}
            selection_done = True

        choices = results["selection"][ds["id"]]
        archive_path = out / f"test_{ds['id']}.npz"
        arrays = dict(np.load(archive_path)) if archive_path.exists() else {}
        arrays["payoff"] = payoff["test"]
        evaluations = results["evaluations"].setdefault(ds["id"], {})

        def evaluate(key, theta, condition, record_id, extra=None):
            if key in evaluations and key in arrays:
                return
            tick = time.perf_counter()
            metrics = net_errors(theta, spot["test"], times, rate, payoff["test"], premium, CONDITIONS[condition])
            errors = metrics.pop("errors")
            if not np.isfinite(errors).all():
                raise ValueError(f"Nonfinite test errors in {key}")
            arrays[key] = errors
            evaluations[key] = dict(fit_id=record_id, condition=condition, metrics=metrics,
                                    evaluation_s=time.perf_counter() - tick, **(extra or {}))
            save()

        # Reference: no trading at all.
        for condition in CONDITIONS:
            evaluate(f"none_{condition}", np.zeros_like(spot["test"][:, 1:]), condition, None, dict(method="none"))

        for method in FROZEN:
            seeds = [0] if method == "affine" else ds["final_seeds"][method]
            for seed in seeds:
                needed_scales = sorted({choices[c][method]["candidate"].get("weight_scale") for c in CONDITIONS}, key=str)
                for s in needed_scales:
                    conditions = [c for c in CONDITIONS if choices[c][method]["candidate"].get("weight_scale") == s]
                    pending = any(f"{method}_{c}_d{seed}" not in evaluations or f"{method}_{c}_d{seed}" not in arrays
                                  for c in conditions)
                    if not pending:
                        continue
                    candidate = choices[conditions[0]][method]["candidate"]
                    phi, psi, scale = group_matrices(method, candidate, seed, "final", ("train", "val", "test"))
                    ridges = {}
                    for condition in conditions:
                        index = choices[condition][method]["candidate_index"]
                        lam = choices[condition][method]["candidate"]["lam"]
                        key = f"{ds['id']}_final_{method}_c{index}_d{seed}"
                        if key not in ridges:
                            ridges[key] = ridge_fit(key, "final", method, index, seed, lam, phi, psi, scale)
                        record = ridges[key] if condition == "zero" else positive_fits(
                            key, "final", method, index, seed, lam, condition, phi, scale, ridges[key], plan["starts"]["final"])
                        theta = readout_positions(phi["test"], np.asarray(record["beta"]), spot["test"])
                        evaluate(f"{method}_{condition}_d{seed}", theta, condition, record["id"],
                                 dict(method=method, seed=seed, candidate_index=index,
                                      start=record.get("start"), optimizer_status=record["optimizer_status"],
                                      best_step=record.get("best_step")))
                    del phi, psi
            print(f"[{task}] final {method} done {ds['id']}", flush=True)

        for condition in (CONDITIONS if ds["partition"] in plan.get("transformer_partitions", [1, 2, 3]) else []):
            key = f"transformer_{condition}"
            if key in evaluations and key in arrays:
                continue
            config_fields = {k: v for k, v in plan["transformer"].items() if k not in ("head_penalty_lam", "seeds")}
            config = TransformerHedgeConfig(**{**config_fields, "seed": ds["transformer_seed"]})
            costs = CostSpec(CONDITIONS[condition], 0.0)
            state_path = out / f"{ds['id']}_transformer_{condition}.pt"
            holder = {}
            def train():
                fitted = fit_cost_transformer(data["train"]["features"], spot["train"], payoff["train"],
                    data["val"]["features"], spot["val"], payoff["val"], times, rate, premium, config,
                    costs=costs, lam=plan["transformer"]["head_penalty_lam"])
                holder["fitted"] = fitted
                torch.save(fitted.model.state_dict(), state_path)
                return dict(validation_mse=fitted.validation_mse, best_epoch=fitted.best_epoch,
                            optimizer_status=fitted.status, gradient_inf=fitted.final_gradient_inf,
                            timings=fitted.timings, n_parameters=fitted.n_parameters,
                            input_mean=fitted.input_mean.tolist(), input_std=fitted.input_std.tolist(),
                            history=[fitted.history[0], fitted.history[-1]])
            record = fit_record(f"{ds['id']}_transformer_{condition}", train, dataset=ds["id"], phase="final",
                                method="transformer", condition=condition, seed=ds["transformer_seed"])
            if "fitted" in holder:
                fitted = holder["fitted"]
            else:  # resume: rebuild the saved network for evaluation only
                n_assets = 1 if spot["train"].ndim == 2 else spot["train"].shape[2]
                network = _build_model(torch, torch.nn, channels, n_assets, config, premium)
                network.load_state_dict(torch.load(state_path, weights_only=True))
                fitted = CostTransformerFit(network, np.asarray(record["input_mean"]), np.asarray(record["input_std"]),
                    config, premium, costs, record["validation_mse"], record["best_epoch"], record["history"],
                    record["optimizer_status"], record["gradient_inf"], record["timings"], record["n_parameters"])
            output = evaluate_cost_transformer(fitted, data["test"]["features"], spot["test"], payoff["test"], times, rate)
            if not np.isfinite(output["errors"]).all():
                raise ValueError("Nonfinite Transformer test errors")
            arrays[key] = output["errors"]
            evaluations[key] = dict(fit_id=record["id"], condition=condition, metrics=output["metrics"],
                                    method="transformer", seed=ds["transformer_seed"],
                                    optimizer_status=record["optimizer_status"], best_epoch=record["best_epoch"])
            save()
            print(f"[{task}] transformer {condition} done {ds['id']}", flush=True)

        temporary = archive_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(archive_path)
        evaluations["completed"] = True
        save()
        print(f"[{task}] partition complete {ds['id']} ({time.perf_counter() - started:.0f}s so far)", flush=True)
        del data, spot, payoff, arrays

    results["status"] = "completed" if not results["failures"] else "completed_with_failures"
    invocation["wall_s"] = time.perf_counter() - started
    save()
    print(f"[{task}] {results['status']}: {len(results['fits'])} fits, {invocation['wall_s']:.0f}s", flush=True)


# --------------------------------------------------------------------------- report
LABELS = {"none": "No hedge", "affine": "Affine", "static": "Static tanh", "leaky": "Leaky tanh",
          "reservoir": "Reservoir", "transformer": "Transformer"}
TITLES = {"black_scholes": "Black–Scholes", "heston": "Heston", "basket": "Heston basket",
          "two_factor_heston": "Two-factor Heston", "volterra": "Volterra H=0.1"}


def report(out, plan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, paired, selections, fit_rows = [], [], [], []
    per_task = {}
    for task in plan["tasks"]:
        path = out / task / "results.json"
        if not path.exists():
            print(f"missing {task}", flush=True)
            continue
        results = json.loads(path.read_text())
        per_task[task] = results
        datasets = [d for d in plan["datasets"] if d["task"] == task]
        for ds in datasets:
            archive_path = out / task / f"test_{ds['id']}.npz"
            if not archive_path.exists():
                continue
            archive = np.load(archive_path)
            evals = results["evaluations"][ds["id"]]
            path_mse = {}
            for condition in CONDITIONS:
                for method in ["none"] + METHODS:
                    keys = [k for k, v in evals.items() if k != "completed" and v["method"] == method and v["condition"] == condition]
                    if not keys:
                        continue
                    errors = np.stack([archive[k] for k in keys])
                    mse_draws = np.mean(errors**2, axis=1)
                    path_mse[(method, condition)] = np.mean(errors**2, axis=0)
                    metric = lambda name: float(np.mean([evals[k]["metrics"][name] for k in keys]))
                    rows.append(dict(task=task, dataset=ds["id"], condition=condition, kappa=CONDITIONS[condition],
                        method=method, draws=len(keys), mse=float(mse_draws.mean()), rmse=float(np.sqrt(mse_draws.mean())),
                        draw_mse_sd=float(np.std(mse_draws, ddof=1)) if len(keys) > 1 else 0.0,
                        mean_error=metric("mean_error"), error_std=metric("error_std"), cvar95=metric("shortfall_cvar95"),
                        turnover=metric("mean_turnover"), cost=metric("mean_cost"),
                        starts=",".join(str(evals[k].get("start")) for k in keys),
                        statuses=",".join(str(evals[k].get("optimizer_status")) for k in keys)))
                for left, right in combinations(["none"] + METHODS, 2):
                    if (left, condition) in path_mse and (right, condition) in path_mse:
                        delta = path_mse[(left, condition)] - path_mse[(right, condition)]
                        paired.append(dict(task=task, dataset=ds["id"], condition=condition, comparison=f"{left}_minus_{right}",
                            mean_difference=float(delta.mean()), se=float(delta.std(ddof=1) / np.sqrt(len(delta))),
                            top5_signed_share=float(np.sort(delta)[::-1][:5].sum() / abs(delta.sum())) if delta.sum() else 0.0))
            for condition, choices in results["selection"].get(ds["id"], {}).items():
                for method, choice in choices.items():
                    selections.append(dict(task=task, dataset=ds["id"], condition=condition, method=method,
                                           **{k: v for k, v in choice["candidate"].items()}))
        for record in results["fits"].values():
            fit_rows.append(dict(task=task, phase=record.get("phase"), method=record.get("method"), condition=record.get("condition"),
                                 status=record.get("optimizer_status"), total_s=record.get("total_s", 0.0), best_step=record.get("best_step")))

    summary = []
    for task in per_task:
        for condition in CONDITIONS:
            for method in ["none"] + METHODS:
                sub = [r for r in rows if (r["task"], r["condition"], r["method"]) == (task, condition, method)]
                if not sub:
                    continue
                mses = [r["mse"] for r in sub]
                summary.append(dict(task=task, condition=condition, kappa=CONDITIONS[condition], method=method, partitions=len(sub),
                    mse=float(np.mean(mses)), rmse=float(np.sqrt(np.mean(mses))), partition_mse_sd=float(np.std(mses, ddof=1)) if len(sub) > 1 else 0.0,
                    mean_error=float(np.mean([r["mean_error"] for r in sub])), error_std=float(np.mean([r["error_std"] for r in sub])),
                    cvar95=float(np.mean([r["cvar95"] for r in sub])), turnover=float(np.mean([r["turnover"] for r in sub])),
                    cost=float(np.mean([r["cost"] for r in sub]))))
    paired_summary = []
    for task in per_task:
        for condition in CONDITIONS:
            for comparison in sorted({p["comparison"] for p in paired}):
                sub = [p for p in paired if (p["task"], p["condition"], p["comparison"]) == (task, condition, comparison)]
                if not sub:
                    continue
                values = [p["mean_difference"] for p in sub]
                paired_summary.append(dict(task=task, condition=condition, comparison=comparison, mean=float(np.mean(values)),
                    partition_sd=float(np.std(values, ddof=1)) if len(sub) > 1 else 0.0,
                    wins=int(sum(v < 0 for v in values)), partitions=len(sub),
                    min_abs_z=float(min(abs(p["mean_difference"]) / p["se"] for p in sub if p["se"] > 0) if sub else 0.0)))
    _write_csv(out / "dataset_summary.csv", rows)
    _write_csv(out / "summary.csv", summary)
    _write_csv(out / "paired.csv", paired)
    _write_csv(out / "paired_summary.csv", paired_summary)
    _write_csv(out / "selections.csv", selections)

    # Figure: pooled RMSE against cost level, one panel per task.
    tasks = [t for t in plan["tasks"] if t in per_task]
    fig, axes = plt.subplots(2, len(tasks), figsize=(3.6 * len(tasks), 7.2), layout="constrained", squeeze=False)
    colors = {"none": "#999999", "affine": "#777777", "static": "#d88a22", "leaky": "#8c564b", "reservoir": "#176b87", "transformer": "#2ca02c"}
    x = np.arange(len(CONDITIONS))
    for ax, ax2, task in zip(axes[0], axes[1], tasks):
        for method in ["none"] + METHODS:
            sub = [next((s for s in summary if (s["task"], s["condition"], s["method"]) == (task, c, method)), None) for c in CONDITIONS]
            if any(s is None for s in sub):
                continue
            ax.plot(x, [s["rmse"] for s in sub], marker="o", color=colors[method], label=LABELS[method])
            ax2.plot(x, [s["mean_error"] for s in sub], marker="o", color=colors[method], label=LABELS[method])
        ax.set(title=TITLES.get(task, task), xticks=x, xticklabels=["0", "10bp", "50bp", "200bp"], ylabel="Test net RMSE")
        ax2.set(xticks=x, xticklabels=["0", "10bp", "50bp", "200bp"], ylabel="Mean net error (cost drag)", xlabel="Proportional cost")
        ax2.axhline(0, color="gray", lw=.8, ls="--")
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Cost-aware training with fixed premium: pooled test RMSE and mean error (3 partitions × 3 draws)")
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)

    # Paired figure: each history representation minus static tanh, by cost.
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 4.2), layout="constrained", squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for j, method in enumerate(["leaky", "reservoir", "transformer"]):
            for i, condition in enumerate(CONDITIONS):
                sub = [p for p in paired if (p["task"], p["condition"], p["comparison"]) == (task, condition, f"static_minus_{method}")]
                if not sub:
                    continue
                values = np.array([p["mean_difference"] for p in sub])
                xx = i + (j - 1) * 0.22
                ax.scatter(np.full(len(values), xx) + np.linspace(-.05, .05, len(values)), values, color=colors[method], s=18, zorder=3,
                           label=LABELS[method] if i == 0 else None)
                ax.plot([xx - .08, xx + .08], [values.mean()] * 2, color="black", lw=1.5)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set(title=TITLES.get(task, task), xticks=x, xticklabels=["0", "10bp", "50bp", "200bp"],
               ylabel="Static MSE − method MSE (positive favors history)")
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Paired differences on shared test paths (dots: partitions; bars: means)")
    fig.savefig(out / "paired.png", dpi=160)
    plt.close(fig)

    lines = ["# Cost-aware memory comparison", "",
             f"Tasks completed: {', '.join(tasks)}. Conditions: proportional cost 0 / 10 / 50 / 200 bp of traded notional; fixed per-partition premium; "
             f"3 partitions × 3 fresh draws per selected setting (affine: one refit; Transformer: one seed, partition(s) {plan.get('transformer_partitions', [1, 2, 3])} only).", "",
             "## Pooled test net RMSE (currency units)", "",
             "| Task | Method | " + " | ".join(CONDITIONS) + " |", "|---|---|" + "---:|" * len(CONDITIONS)]
    for task in tasks:
        for method in ["none"] + METHODS:
            cells = []
            for condition in CONDITIONS:
                s = next((s for s in summary if (s["task"], s["condition"], s["method"]) == (task, condition, method)), None)
                cells.append(f"{s['rmse']:.4f}" if s else "–")
            lines.append(f"| {TITLES.get(task, task)} | {LABELS[method]} | " + " | ".join(cells) + " |")
    lines += ["", "## Mean net error (cost drag at the frictionless premium) and mean turnover (shares)", "",
              "| Task | Method | " + " | ".join(f"{c} error / turnover" for c in CONDITIONS) + " |", "|---|---|" + "---:|" * len(CONDITIONS)]
    for task in tasks:
        for method in ["none"] + METHODS:
            cells = []
            for condition in CONDITIONS:
                s = next((s for s in summary if (s["task"], s["condition"], s["method"]) == (task, condition, method)), None)
                cells.append(f"{s['mean_error']:+.3f} / {s['turnover']:.2f}" if s else "–")
            lines.append(f"| {TITLES.get(task, task)} | {LABELS[method]} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired MSE differences against static tanh (mean ± partition SD; positive favors the history method; wins = partitions where the history method has lower MSE)", "",
              "| Task | Comparison | " + " | ".join(CONDITIONS) + " |", "|---|---|" + "---:|" * len(CONDITIONS)]
    for task in tasks:
        for method in ["leaky", "reservoir", "transformer"]:
            cells = []
            for condition in CONDITIONS:
                p = next((p for p in paired_summary if (p["task"], p["condition"], p["comparison"]) == (task, condition, f"static_minus_{method}")), None)
                cells.append(f"{p['mean']:+.4f} ± {p['partition_sd']:.4f} ({p['partitions'] - p['wins']}/{p['partitions']})" if p else "–")
            lines.append(f"| {TITLES.get(task, task)} | static − {LABELS[method].lower()} | " + " | ".join(cells) + " |")
    lines += ["", "## Selected settings (weight scale s, ridge penalty λ) by condition", ""]
    for task in tasks:
        for method in FROZEN:
            parts = []
            for condition in CONDITIONS:
                sub = [s for s in selections if (s["task"], s["condition"], s["method"]) == (task, condition, method)]
                parts.append(condition + ": " + ", ".join(
                    (f"(s={s['weight_scale']:g}, λ={s['lam']:g})" if "weight_scale" in s else f"λ={s['lam']:g}") for s in sub))
            lines.append(f"- {TITLES.get(task, task)}, {LABELS[method]}: " + "; ".join(parts))
    from collections import Counter
    statuses = Counter(r["status"] for r in fit_rows)
    fit_time = sum(r["total_s"] or 0.0 for r in fit_rows)
    lines += ["", f"Fits: {len(fit_rows)} counted; optimizer outcomes {dict(statuses)}; total fit time {fit_time:.0f}s "
              f"(per-task walls: {', '.join(f'{t}: {sum(i.get('wall_s', 0.) for i in r['invocations']):.0f}s' for t, r in per_task.items())}).",
              "", "![RMSE and cost drag](comparison.png)", "", "![Paired differences](paired.png)", "",
              "Positive mean error is the expected cost at a premium that excludes it; the hedge is still selected and trained on net MSE, which includes that drag. "
              "'No hedge' pays nothing and bears the full payoff variance. Positive-cost readouts start from the zero-cost ridge solution and run L-BFGS on the "
              "smoothed-cost training objective for at most 200 iterations ('budget_exhausted' = cap reached, 'lbfgs_terminated' = line search or tolerance stop, "
              "'stationary' = gradient tolerance met); none of these is a global-optimality certificate. All methods are open-loop in the position (no feedback of the "
              "held position into the policy), so the comparison tests whether history *features* can substitute for position feedback under costs.", "",
              "Execution notes: the run was launched twice before its final form (two tuning draws, then Transformer on all partitions); each time it was stopped after timing, "
              "re-declared, and resumed with completed fits reused, so a few unused draw-2 tuning records and interrupted 'running' records remain in results.json and in the "
              "fit count above. One restart happened after partition-1 evaluations were recorded but before their error arrays were archived; those arrays were recomputed from the "
              "saved coefficients (no refits) by a repair pass. Fit times are from five concurrent, memory-bandwidth-bound processes and are not a controlled benchmark."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:60]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--declare", action="store_true")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny integration run; separate output folder")
    args = parser.parse_args()
    plan = PLAN
    out = OUT
    if args.smoke:
        out = OUT / "smoke"
        plan = json.loads(json.dumps(PLAN))
        plan["sizes"] = {"train": 120, "val": 60, "test": 100}
        plan["n_steps"] = 6
        plan["optimizer"] = dict(steps=3, lr=0.01, grad_tol=1e-5)
        plan["candidates"] = [dict(dim=4, weight_scale=1.0, bias_scale=1.0, lam=l) for l in (0.1, 10.0)]
        plan["affine_candidates"] = [dict(lam=l) for l in (0.1, 10.0)]
        plan["transformer"] = {**plan["transformer"], "d_model": 8, "n_heads": 2, "n_layers": 1, "epochs": 1, "window": 3}
        for ds in plan["datasets"]:
            for group in ("tuning_seeds", "final_seeds"):
                ds[group] = {m: v[:2] for m, v in ds[group].items()}
        plan["datasets"] = [d for d in plan["datasets"] if d["partition"] == 1]
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if args.declare or args.smoke:
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise RuntimeError("Existing declaration differs; refusing overwrite.")
        if not plan_path.exists():
            sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), *sorted((ROOT / "rsmc").glob("*.py"))]}
            write_json(plan_path, plan)
            write_json(out / "declaration.json", dict(declared_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources))
            print(f"Declared {len(plan['datasets'])} datasets, {len(METHODS)} methods, {len(CONDITIONS)} conditions -> {plan_path}")
        if args.declare:
            return
    if not plan_path.exists() or json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Declare first; code differs from the saved plan.")
    if args.report:
        report(out, plan)
        return
    tasks = [args.task] if args.task else TASKS
    for task in tasks:
        run_task(task, out / task, plan, smoke=args.smoke)
    if args.smoke:
        report(out, plan)


if __name__ == "__main__":
    main()
