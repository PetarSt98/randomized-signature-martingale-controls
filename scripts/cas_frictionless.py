"""Bounded, predeclared Heston Asian hedge comparison; no noise or filtering.

Declare first with --declare, then run without arguments. Repeating the run
resumes completed records without refitting them. Outputs live in results/.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsmc.heston import HestonModel, make_heston_datasets
from rsmc.payoffs import arithmetic_asian_call
from rsmc.hedging import (
    hedging_features, discounted_traded_increments,
    assemble_linear_hedge_features, assemble_reservoir_hedge_features,
    fit_fixed_premium_hedge, evaluate_hedge,
)
from rsmc.reservoir import RandomizedSignatureReservoir, DepthTwoSignatureReservoir
from rsmc.transformer_hedging import (
    TransformerHedgeConfig, fit_transformer_hedge, evaluate_transformer_hedge,
)

OUT = ROOT / "results" / "cas_frictionless"
# Sparse matched lists, deliberately not a factorial search. Never expand after running.
TUPLES = [(16, .5, .1), (16, 1., 10.), (32, .5, 10.),
          (32, 1., .1), (64, .5, .1), (64, 1., 10.)]
PLAN = {
    "model": asdict(HestonModel()), "strike": 100., "n_steps": 50,
    "sizes": {"train": 4000, "val": 2000, "test": 8000},
    "dataset_seed": 20261027,
    "information": ["time", "log_price", "running_average"],
    "premium_rule": "float32-rounded discounted training payoff mean; same frozen value on every split and method; reuse for later costs",
    "validation_draw_seeds": [31001, 31002, 31003],
    "fresh_draw_seeds": [41001, 41002, 41003],
    "candidates": [dict(architecture=a, dim=d, weight_scale=s, bias_scale=1., lam=l)
                   for a in ("shallow", "depth_two") for d, s, l in TUPLES],
    "selection": "lowest average unpenalized validation MSE over all three draws; first tuple breaks ties; incomplete tuples ineligible",
    "simple_hedge_lambda": .1,
    "transformer": asdict(TransformerHedgeConfig(seed=51001)),
    "fit_budget": {"reservoir_tuning": 36, "fresh_reservoir": 6,
                   "simple_hedge": 1, "transformer_training_runs": 1, "total": 44},
    "test_policy": "selected configurations only, fresh reservoir draws fitted on training paths; test paths independent of train/validation; no recentering",
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def metrics(report):
    names = ("n_eval", "price", "capital", "rmse", "error_mse", "error_var",
             "mean_error", "mean_error_se", "variance_reduction")
    result = {key: getattr(report, key) for key in names}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite report; retain as failed fit, never select it.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plan_file = OUT / "plan.json"
    if args.declare:
        if plan_file.exists() and json.loads(plan_file.read_text()) != PLAN:
            raise RuntimeError("Existing declaration differs; refusing to overwrite it.")
        if not plan_file.exists():
            write_json(plan_file, PLAN)
        print(f"Declared 12 tuples, 36 tuning fits, 44 total fits: {plan_file}")
        return
    plan = json.loads(plan_file.read_text())
    if plan != PLAN:
        raise RuntimeError("Code and saved declaration differ; refusing an altered search.")
    digest = hashlib.sha256(plan_file.read_bytes()).hexdigest()
    result_file = OUT / "results.json"
    results = json.loads(result_file.read_text()) if result_file.exists() else {
        "status": "running", "plan_sha256": digest, "fits": [], "failures": []}
    if results["plan_sha256"] != digest:
        raise RuntimeError("Results belong to a different declaration.")
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__), *sorted((ROOT / "rsmc").glob("*.py"))]}
    if "source_sha256" in results and results["source_sha256"] != source_hashes:
        raise RuntimeError("Implementation changed since the run; refusing to mix results.")
    results["source_sha256"] = source_hashes
    # Regeneration is deterministic and does not fit anything on resume.
    start = time.perf_counter()
    model = HestonModel(**plan["model"])
    paths = make_heston_datasets(model, plan["sizes"], plan["n_steps"], plan["dataset_seed"])
    bundles = {}
    for name, p in paths.items():
        bundles[name] = (hedging_features(p.spot, p.times), p.times,
                         discounted_traded_increments(p.spot, p.times, model.r),
                         arithmetic_asian_call(p, plan["strike"], model.r))
    premium = float(np.float32(bundles["train"][3].mean()))
    if "premium" in results and results["premium"] != premium:
        raise RuntimeError("Frozen premium changed; refusing to resume.")
    results.setdefault("data_and_state_features_s", time.perf_counter() - start)
    results["premium"] = premium
    results["premium_training_standard_error"] = float(bundles["train"][3].std(ddof=1) / np.sqrt(plan["sizes"]["train"]))
    dataset_hashes = {k: hashlib.sha256(p.spot.tobytes()).hexdigest() for k, p in paths.items()}
    if "dataset_spot_sha256" in results and results["dataset_spot_sha256"] != dataset_hashes:
        raise RuntimeError("Dataset changed since the run; refusing to mix results.")
    results["dataset_spot_sha256"] = dataset_hashes
    import torch
    import scipy
    results["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                           "scipy": scipy.__version__, "torch": torch.__version__}
    results["threads"] = {"torch": torch.get_num_threads(), **{
        name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    results["platform"] = platform.platform()
    write_json(result_file, results)

    def checkpoint():
        write_json(result_file, results)

    def execute(key, phase, method, fn, **metadata):
        old = next((r for r in results["fits"] if r["id"] == key), None)
        if old is not None:
            if old["status"] == "running":
                old["status"] = "failed"
                old["failure"] = "Interrupted attempt; counted conservatively, not repeated."
                results["failures"].append({"id": key, "error": old["failure"]})
                checkpoint()
            return old
        if len(results["fits"]) >= plan["fit_budget"]["total"]:
            raise RuntimeError("Predeclared fit budget exhausted.")
        record = dict(id=key, phase=phase, method=method, status="running", **metadata)
        results["fits"].append(record)
        checkpoint()  # Count the attempt before work; never silently repeat a fit.
        started = time.perf_counter()
        try:
            record.update(fn(record))
            record["status"] = "completed"
        except Exception:
            record["status"] = "failed"
            record["failure"] = traceback.format_exc()
            results["failures"].append({"id": key, "error": record["failure"]})
        record["total_s"] = time.perf_counter() - started
        checkpoint()
        print(f"{len(results['fits']):2d}/44 {key}: {record['status']} "
              f"({record['total_s']:.2f}s)", flush=True)
        return record

    def ridge_run(record, candidate=None, seed=None, test=False):
        if candidate is None:
            assemble = assemble_linear_hedge_features
            lam = plan["simple_hedge_lambda"]
        else:
            cls = RandomizedSignatureReservoir if candidate["architecture"] == "shallow" else DepthTwoSignatureReservoir
            reservoir = cls.draw(3, candidate["dim"], np.random.default_rng(seed),
                                 weight_scale=candidate["weight_scale"], bias_scale=candidate["bias_scale"])
            assemble = lambda f, t, inc: assemble_reservoir_hedge_features(reservoir, f, t, inc, batch_size=1000)
            lam = candidate["lam"]
        matrices, feature_times = {}, {}
        splits = ("train", "test") if test else ("train", "val")
        for split in splits:
            matrices[split], feature_times[split] = timed(lambda: assemble(*bundles[split][:3]))
        control, fit_s = timed(lambda: fit_fixed_premium_hedge(matrices["train"], bundles["train"][3], premium, lam))
        split = splits[-1]
        report, evaluation_s = timed(lambda: evaluate_hedge(control, matrices[split], bundles[split][3]))
        values = metrics(report)
        if test:
            np.save(OUT / (record["id"] + "_errors.npy"), report.errors)
        return {"metrics": values, "feature_generation_s": feature_times, "fit_s": fit_s,
                "evaluation_s": evaluation_s, "beta": control.beta.tolist(),
                "lambda": lam, "readout_parameters": len(control.beta)}

    # No test outcomes are computed until all 36 selection attempts are complete.
    for index, candidate in enumerate(plan["candidates"]):
        for seed in plan["validation_draw_seeds"]:
            execute(f"tune_{index:02d}_{seed}", "tuning", candidate["architecture"],
                    lambda r: ridge_run(r, candidate, seed), candidate_index=index, seed=seed)
    selected = {}
    for method in ("shallow", "depth_two"):
        choices = []
        for index, candidate in enumerate(plan["candidates"]):
            if candidate["architecture"] != method:
                continue
            records = [r for r in results["fits"] if r.get("candidate_index") == index and r["phase"] == "tuning"]
            if len(records) == 3 and all(r["status"] == "completed" for r in records):
                choices.append((float(np.mean([r["metrics"]["error_mse"] for r in records])), index))
        if not choices:
            raise RuntimeError(f"No complete eligible candidate for {method}; inspect failures.")
        score, index = min(choices)
        selected[method] = dict(candidate_index=index, candidate=plan["candidates"][index], validation_mse=score)
    if "selected" in results and results["selected"] != selected:
        raise RuntimeError("Selection changed on resume.")
    results["selected"] = selected
    checkpoint()
    for method, choice in selected.items():
        for seed in plan["fresh_draw_seeds"]:
            execute(f"final_{method}_{seed}", "test", method,
                    lambda r: ridge_run(r, choice["candidate"], seed, test=True),
                    candidate_index=choice["candidate_index"], seed=seed)
    execute("final_simple", "test", "simple", lambda r: ridge_run(r, test=True))

    def transformer_run(record):
        config = TransformerHedgeConfig(**plan["transformer"])
        train, val, test = bundles["train"], bundles["val"], bundles["test"]
        fitted, fit_s = timed(lambda: fit_transformer_hedge(
            train[0], train[2], train[3], val[0], val[2], val[3], config, initial_capital=premium))
        report, evaluation_s = timed(lambda: evaluate_transformer_hedge(fitted, test[0], test[2], test[3]))
        values = metrics(report)
        np.save(OUT / (record["id"] + "_errors.npy"), report.errors)
        torch.save(fitted.model.state_dict(), OUT / "transformer_state.pt")
        return {"metrics": values, "feature_generation_s": {}, "fit_s": fit_s,
                "evaluation_s": evaluation_s, "validation_mse": fitted.validation_mse,
                "input_mean": fitted.input_mean.tolist(), "input_std": fitted.input_std.tolist(),
                "readout_parameters": fitted.n_parameters, "epochs": config.epochs,
                "optimizer_steps": config.epochs * int(np.ceil(plan["sizes"]["train"] / config.batch_size)),
                "timing_note": "fit includes input normalization, model construction, training and epoch validation; evaluation includes input transformation"}
    execute("final_transformer", "test", "transformer", transformer_run, seed=plan["transformer"]["seed"])
    results["status"] = "completed" if len(results["fits"]) == 44 and not results["failures"] else "completed_with_failures"
    results["fit_attempts"] = len(results["fits"])
    checkpoint()
    render(results)


def render(results):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"simple": "Simple state hedge", "shallow": "Shallow tanh", "depth_two": "Depth-two tanh", "transformer": "Transformer"}
    summary = []
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for position, (method, label) in enumerate(labels.items()):
        final = [r for r in results["fits"] if r["method"] == method and r["phase"] == "test" and r["status"] == "completed"]
        if not final:
            continue
        tuning = [r for r in results["fits"] if r["method"] == method and r["phase"] == "tuning"]
        mse = [r["metrics"]["error_mse"] for r in final]
        row = dict(method=method, draws=len(final), premium=results["premium"],
                   test_mse=float(np.mean(mse)), test_rmse=float(np.sqrt(np.mean(mse))),
                   draw_mse_sd=float(np.std(mse, ddof=1)) if len(mse) > 1 else 0.,
                   mean_error=float(np.mean([r["metrics"]["mean_error"] for r in final])),
                   tuning_s=sum(r["total_s"] for r in tuning),
                   final_feature_s=sum(sum(r["feature_generation_s"].values()) for r in final),
                   final_fit_s=sum(r["fit_s"] for r in final),
                   final_evaluation_s=sum(r["evaluation_s"] for r in final))
        summary.append(row)
        ax.bar(position, row["test_rmse"], width=.65, color="#497caa")
        ax.scatter(np.linspace(position-.12, position+.12, len(final)), np.sqrt(mse), color="#202020", s=23, zorder=3)
    if not summary:
        plt.close(fig)
        (OUT / "REPORT.md").write_text("No final evaluation succeeded. See results.json for every failure and counted attempt.\n")
        return
    ax.set_xticks(range(4), list(labels.values()))
    ax.set_ylabel("Test hedge RMSE (discounted price units)")
    ax.set_title("Heston Asian call · frictionless · common fixed premium")
    ax.text(.01, .97, "Bars: sqrt(mean MSE); dots: individual fresh draws", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_ylim(0, max(row["test_rmse"] for row in summary) * 1.22)
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Bounded frictionless Heston comparison", "",
             f"Status: {results['status']}; fit attempts: {results['fit_attempts']}/44; failures: {len(results['failures'])}.",
             f"Frozen premium: {results['premium']:.9f}; training estimate SE: {results['premium_training_standard_error']:.6f}.", "",
             "| Method | Fresh draws | Test MSE (units²) | RMSE (units) | Mean error (units) | Tuning (s) | Final features / fit / evaluation (s) |",
             "|---|---:|---:|---:|---:|---:|---|",]
    for r in summary:
        lines.append(f"| {labels[r['method']]} | {r['draws']} | {r['test_mse']:.6f} | {r['test_rmse']:.6f} | {r['mean_error']:+.6f} | {r['tuning_s']:.2f} | {r['final_feature_s']:.2f} / {r['final_fit_s']:.2f} / {r['final_evaluation_s']:.2f} |")
    lines += ["", "Times are totals over the listed draws, not per-fit averages. Shared simulation/state features are recorded separately in results.json. Transformer input preprocessing and epoch selection are included in its fit time; evaluation includes input transformation.", "", "Selected configurations:"]
    for method, choice in results["selected"].items():
        lines.append(f"- {method}: {choice['candidate']}; three-draw mean validation MSE {choice['validation_mse']:.6f}.")
    lines += ["", "![Comparison](comparison.png)", "",
              "One dataset partition, three validation and three fresh reservoir draws per architecture; one Transformer seed. Sparse non-factorial candidates; equal state/readout width but depth-two has more frozen parameters and computation. No transaction/execution costs, transfer study, hidden variance inputs, synthetic signals or filtering. The finite-sample premium is frozen, so mean error is retained. No significance or universal architecture-ranking claim follows from this run.", "",
              "Declaration: plan.json. Every fit, failure, seed, coefficient, metric and timing: results.json. Test residual arrays and the Transformer state are retained alongside this report.", "",
              "Run/resume from the repository root:", "```bash",
              "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/workspace/.cache/rsmc-matplotlib /workspace/.venvs/rsmc/bin/python scripts/cas_frictionless.py", "```"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:11]), flush=True)


if __name__ == "__main__":
    main()
