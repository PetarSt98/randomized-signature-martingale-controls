"""Bounded cost-aware extension of the saved CAS reference; declare, then run.

Completed fit records are reused on resume. All planned comparisons must exist
before completion is reported; identical configuration references share fits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from cas_frictionless import ROOT, PLAN as BASE, write_json, timed
from rsmc.heston import HestonModel, make_heston_datasets
from rsmc.hedging import hedging_features, fit_fixed_premium_hedge
from rsmc.payoffs import arithmetic_asian_call
from rsmc.reservoir import RandomizedSignatureReservoir, DepthTwoSignatureReservoir
from rsmc.cost_hedging import (
    CostSpec, execution_cashflows, cost_metrics, hedge_readout_features,
    gain_row_scale, fit_cost_readout,
)
from rsmc.cost_transformer import (
    CostTransformerFit, fit_cost_transformer, evaluate_cost_transformer,
)
from rsmc.transformer_hedging import TransformerHedgeConfig, _build_model

OUT = ROOT / "results" / "cas_costs"
OLD = ROOT / "results" / "cas_frictionless"
CONDITIONS = {"zero": CostSpec(0., 0.), "proportional": CostSpec(.0005, 0.),
              "combined": CostSpec(.0005, .0005)}
PREMIUM = 5.27387809753418
PLAN = {
    "reference": BASE,
    "premium": PREMIUM,
    "conditions": {k: asdict(v) for k, v in CONDITIONS.items()},
    "cost_units": "Illustrative uncalibrated dimensionless per-execution rates; q_ref=1 share; costs in discounted price units. q=(theta0,diff(theta),-theta_last); sum exp(-rt)*S*(kappa*abs(q)+eta*q^2/q_ref). No dt divisor; no price impact on future market paths.",
    "readout_optimizer": {"steps": 120, "lr": .01, "grad_tol": 1e-5},
    "readout_restarts": ["new zero-cost ridge initializer", "same initializer plus N(0,0.01^2) in RMS-scaled coordinates"],
    "restart_seed_rule": "61000 + reservoir_draw_seed + 100*candidate_index (simple uses index 12)",
    "penalty": "net MSE + lambda/n_train * squared RMS-scaled readout coefficients; Transformer lambda=0.1 on head weights+bias; AdamW weight_decay=0 throughout",
    "selection": "minimum unpenalized validation net MSE across checkpoints and two starts, averaged across the three validation reservoir draws; first tuple breaks ties",
    "transformer_seeds": {"zero": [51001], "proportional": [51001, 51002], "combined": [51001, 51002]},
    "budget": {"ridge_solves_max": 55, "positive_reservoir_tuning": 144,
               "positive_reservoir_final_max": 48, "positive_simple": 4,
               "transformer_runs": 5, "total_max": 256},
    "comparison_policy": "For each architecture and positive cost, compare zero-selected and cost-selected tuples, retraining both on that cost with three fresh reservoir draws and two starts. Identical tuples alias the same fits explicitly. Recompute all zero results; no historical metric substitution.",
    "scope": "Implemented policy classes only; no global optimum, price feedback, new input, or new architecture.",
}


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def main():
    invocation_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.declare:
        target = OUT / "plan.json"
        if target.exists() and json.loads(target.read_text()) != PLAN:
            raise RuntimeError("Existing declaration differs; refusing to overwrite.")
        if not target.exists():
            write_json(target, PLAN)
        print("Declared three conditions, unchanged 12-tuples, 3 reservoir draws, <=256 counted fits.")
        return
    if json.loads((OUT / "plan.json").read_text()) != PLAN:
        raise RuntimeError("Saved declaration differs from code.")
    old = json.loads((OLD / "results.json").read_text())
    assert old["premium"] == PREMIUM
    result_file = OUT / "results.json"
    results = json.loads(result_file.read_text()) if result_file.exists() else {
        "status": "running", "fits": {}, "feature_groups": {}, "evaluations": {},
        "failures": [], "premium": PREMIUM,
    }
    sources = [Path(__file__), ROOT / "scripts/cas_frictionless.py", *sorted((ROOT / "rsmc").glob("*.py")), OUT / "plan.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if "source_sha256" in results and results["source_sha256"] != hashes:
        raise RuntimeError("Source/declaration changed since this run; refusing mixed results.")
    results["source_sha256"] = hashes
    results["versions"] = {"numpy": np.__version__, "torch": torch.__version__}
    results["torch_threads"] = torch.get_num_threads()
    started = time.perf_counter()
    model = HestonModel(**BASE["model"])
    paths = make_heston_datasets(model, BASE["sizes"], BASE["n_steps"], BASE["dataset_seed"])
    data = {name: dict(features=hedging_features(p.spot, p.times), spot=p.spot,
                       payoff=arithmetic_asian_call(p, BASE["strike"], model.r)) for name, p in paths.items()}
    times = paths["train"].times
    assert float(np.float32(data["train"]["payoff"].mean())) == PREMIUM
    dataset_hashes = {k: hashlib.sha256(p.spot.tobytes()).hexdigest() for k, p in paths.items()}
    if dataset_hashes != old["dataset_spot_sha256"]:
        raise RuntimeError("Reference dataset fingerprints changed.")
    results["dataset_spot_sha256"] = dataset_hashes
    results.setdefault("data_generation_s", time.perf_counter() - started)

    def save():
        write_json(result_file, results)

    def fit_record(key, phase, condition, method, operation, **metadata):
        if key in results["fits"]:
            record = results["fits"][key]
            if record["status"] != "completed":
                raise RuntimeError(f"Unfinished/failed fit {key}; outputs retained. Do not silently repeat beyond its declared budget.")
            return record
        if len(results["fits"]) >= PLAN["budget"]["total_max"]:
            raise RuntimeError("Declared fit cap reached.")
        record = dict(id=key, phase=phase, condition=condition, method=method,
                      status="running", **metadata)
        results["fits"][key] = record
        save()
        start = time.perf_counter()
        try:
            record.update(json_value(operation()))
            record["status"] = "completed"
        except Exception:
            record["status"] = "failed"
            record["failure"] = traceback.format_exc()
            results["failures"].append({"id": key, "error": record["failure"]})
            results["status"] = "incomplete"
            raise
        finally:
            record["total_s"] = time.perf_counter() - start
            save()
            print(f"fit {len(results['fits']):3d}/<=256 {key}: {record['status']} ({record['total_s']:.2f}s)", flush=True)
        return record

    def features(index, seed, phase):
        start = time.perf_counter()
        if index == 12:
            reservoir = None
        else:
            candidate = BASE["candidates"][index]
            cls = RandomizedSignatureReservoir if candidate["architecture"] == "shallow" else DepthTwoSignatureReservoir
            reservoir = cls.draw(3, candidate["dim"], np.random.default_rng(seed),
                                 weight_scale=candidate["weight_scale"], bias_scale=candidate["bias_scale"])
        draw_s = time.perf_counter() - start
        matrices, elapsed = {}, {}
        for split in ("train", "val"):
            matrices[split], elapsed[split] = timed(lambda: hedge_readout_features(data[split]["features"], times, reservoir))
        group = f"{phase}_{index}_{seed}"
        results["feature_groups"].setdefault(group, {"draw_s": draw_s, "splits_s": elapsed, "candidate_index": index, "seed": seed, "phase": phase})
        save()
        return reservoir, matrices, group

    def get_ridge(index, seed, phase, phi):
        method = "simple" if index == 12 else BASE["candidates"][index]["architecture"]
        lam = BASE["simple_hedge_lambda"] if index == 12 else BASE["candidates"][index]["lam"]
        def solve():
            (psi, scale), scale_s = timed(lambda: gain_row_scale(phi["train"], data["train"]["spot"], times, model.r))
            fitted, fit_s = timed(lambda: fit_fixed_premium_hedge(psi, data["train"]["payoff"], PREMIUM, lam))
            theta = phi["val"][:, :-1] @ fitted.beta
            val, validation_s = timed(lambda: cost_metrics(data["val"]["payoff"], PREMIUM, execution_cashflows(theta, data["val"]["spot"], times, model.r, CONDITIONS["zero"])))
            val.pop("errors")
            return dict(beta=fitted.beta, scale=scale, validation_mse=val["net_mse"], validation_metrics=val,
                        optimizer_status="exact_ridge", fit_s=fit_s, setup_s=scale_s, validation_s=validation_s)
        return fit_record(f"ridge_{phase}_{index}_{seed}", phase, "zero", method, solve, candidate_index=index, seed=seed, lam=lam)

    def positive_fits(index, seed, phase, condition, phi, ridge):
        records = []
        lam = ridge["lam"]
        scale = np.asarray(ridge["scale"])
        initial = np.asarray(ridge["beta"]) * scale
        for restart in range(2):
            init = initial.copy()
            if restart:
                init += np.random.default_rng(61000 + seed + 100 * index).normal(0., .01, size=init.shape)
            def optimize():
                fitted = fit_cost_readout(phi["train"], data["train"]["spot"], data["train"]["payoff"],
                    phi["val"], data["val"]["spot"], data["val"]["payoff"], times, model.r,
                    PREMIUM, CONDITIONS[condition], lam, scale, init, **PLAN["readout_optimizer"])
                fitted["optimizer_status"] = fitted.pop("status")
                return fitted
            records.append(fit_record(f"opt_{phase}_{condition}_{index}_{seed}_{restart}", phase, condition,
                                      ridge["method"], optimize, candidate_index=index, seed=seed, restart=restart, lam=lam))
        return min(records, key=lambda r: (r["validation_mse"], r["restart"]))

    # Run every declared candidate/draw/condition before selecting any configuration.
    for index in range(12):
        for seed in BASE["validation_draw_seeds"]:
            _, phi, group = features(index, seed, "tuning")
            ridge = get_ridge(index, seed, "tuning", phi)
            for condition in ("proportional", "combined"):
                positive_fits(index, seed, "tuning", condition, phi, ridge)
            del phi
    selected = {}
    for condition in CONDITIONS:
        selected[condition] = {}
        for method in ("shallow", "depth_two"):
            candidates = []
            for index, candidate in enumerate(BASE["candidates"]):
                if candidate["architecture"] != method:
                    continue
                scores = []
                for seed in BASE["validation_draw_seeds"]:
                    records = [r for r in results["fits"].values() if r["phase"] == "tuning" and r["condition"] == condition and r["candidate_index"] == index and r["seed"] == seed]
                    if len(records) != (1 if condition == "zero" else 2) or any(r["status"] != "completed" for r in records):
                        raise RuntimeError("Incomplete declared selection; cannot omit failed candidates.")
                    scores.append(min(r["validation_mse"] for r in records))
                candidates.append((float(np.mean(scores)), index))
            score, index = min(candidates)
            selected[condition][method] = dict(candidate_index=index, validation_mse=score, candidate=BASE["candidates"][index])
    if "selected" in results and results["selected"] != selected:
        raise RuntimeError("Selections changed on resume.")
    results["selected"] = selected
    results["reference_configuration"] = {c: min(choices.values(), key=lambda x: (x["validation_mse"], x["candidate_index"])) for c, choices in selected.items()}
    save()

    def save_evaluation(key, record, condition, method, reference, evaluate):
        if key in results["evaluations"]:
            return
        start = time.perf_counter()
        metrics, cash = evaluate()
        errors = metrics.pop("errors")
        archive = {k: v for k, v in cash.items() if k != "trades"}
        np.savez_compressed(OUT / f"{key}.npz", errors=errors, **archive)
        results["evaluations"][key] = dict(fit_id=record["id"], condition=condition, method=method,
            reference=reference, seed=record.get("seed"), candidate_index=record.get("candidate_index"),
            metrics=metrics, evaluation_and_save_s=time.perf_counter() - start)
        save()

    for method in ("shallow", "depth_two"):
        zero_index = selected["zero"][method]["candidate_index"]
        indices = sorted({selected[c][method]["candidate_index"] for c in CONDITIONS})
        for index in indices:
            for seed in BASE["fresh_draw_seeds"]:
                reservoir, phi, group = features(index, seed, "fresh")
                ridge = get_ridge(index, seed, "fresh", phi)
                test_phi, feature_s = timed(lambda: hedge_readout_features(data["test"]["features"], times, reservoir))
                results["feature_groups"][group]["splits_s"].setdefault("test", feature_s)
                for condition in CONDITIONS:
                    references = []
                    if index == selected[condition][method]["candidate_index"]:
                        references.append("cost_selected")
                    if condition != "zero" and index == zero_index:
                        references.append("frictionless_selected")
                    if not references:
                        continue
                    record = ridge if condition == "zero" else positive_fits(index, seed, "fresh", condition, phi, ridge)
                    for reference in references:
                        key = f"test_{condition}_{method}_{reference}_{seed}"
                        def evaluate():
                            cash = execution_cashflows(test_phi[:, :-1] @ np.asarray(record["beta"]), data["test"]["spot"], times, model.r, CONDITIONS[condition])
                            return cost_metrics(data["test"]["payoff"], PREMIUM, cash), cash
                        save_evaluation(key, record, condition, method, reference, evaluate)
                del phi, test_phi

    # The same simple hedge and Transformer are fitted under each condition.
    _, phi, group = features(12, 0, "baseline")
    ridge = get_ridge(12, 0, "baseline", phi)
    simple_test, feature_s = timed(lambda: hedge_readout_features(data["test"]["features"], times))
    results["feature_groups"][group]["splits_s"].setdefault("test", feature_s)
    for condition in CONDITIONS:
        record = ridge if condition == "zero" else positive_fits(12, 0, "baseline", condition, phi, ridge)
        def evaluate_simple():
            cash = execution_cashflows(simple_test[:, :-1] @ np.asarray(record["beta"]), data["test"]["spot"], times, model.r, CONDITIONS[condition])
            return cost_metrics(data["test"]["payoff"], PREMIUM, cash), cash
        save_evaluation(f"test_{condition}_simple", record, condition, "simple", "baseline", evaluate_simple)
    del phi, simple_test

    for condition in CONDITIONS:
        records = []
        for seed in PLAN["transformer_seeds"][condition]:
            config = TransformerHedgeConfig(**{**BASE["transformer"], "seed": seed, "weight_decay": 0.})
            key = f"transformer_{condition}_{seed}"
            def train_transformer():
                fitted = fit_cost_transformer(data["train"]["features"], data["train"]["spot"], data["train"]["payoff"],
                    data["val"]["features"], data["val"]["spot"], data["val"]["payoff"], times, model.r,
                    PREMIUM, config, costs=CONDITIONS[condition], lam=.1)
                torch.save(fitted.model.state_dict(), OUT / f"{key}.pt")
                return dict(config=asdict(config), input_mean=fitted.input_mean, input_std=fitted.input_std,
                    validation_mse=fitted.validation_mse, best_epoch=fitted.best_epoch, history=fitted.history,
                    optimizer_status=fitted.status, gradient_inf=fitted.final_gradient_inf,
                    timings=fitted.timings, n_parameters=fitted.n_parameters,
                    optimizer_steps=config.epochs * int(np.ceil(BASE["sizes"]["train"] / config.batch_size)))
            records.append(fit_record(key, "baseline", condition, "transformer", train_transformer, seed=seed, lam=.1))
        record = min(records, key=lambda r: (r["validation_mse"], r["seed"]))
        def evaluate_transformer():
            config = TransformerHedgeConfig(**record["config"])
            network = _build_model(torch, torch.nn, 3, 1, config, PREMIUM)
            network.load_state_dict(torch.load(OUT / f"{record['id']}.pt", weights_only=True))
            fitted = CostTransformerFit(network, np.asarray(record["input_mean"]), np.asarray(record["input_std"]),
                config, PREMIUM, CONDITIONS[condition], record["validation_mse"], record["best_epoch"],
                record["history"], record["optimizer_status"], record["gradient_inf"], record["timings"], record["n_parameters"])
            output = evaluate_cost_transformer(fitted, data["test"]["features"], data["test"]["spot"], data["test"]["payoff"], times, model.r)
            return {**output["metrics"], "errors": output["errors"]}, output["cashflows"]
        save_evaluation(f"test_{condition}_transformer", record, condition, "transformer", "baseline", evaluate_transformer)
    # 6 zero reservoir + 24 positive reservoir (both references) + 6 baselines.
    if len(results["evaluations"]) != 36 or results["failures"]:
        raise RuntimeError("Study incomplete: every declared comparison must finish.")
    results["status"] = "completed"
    results["fit_count"] = len(results["fits"])
    results["aliased_comparisons"] = {key: [k for k, r in results["evaluations"].items() if r["fit_id"] == value["fit_id"]]
        for key, value in results["evaluations"].items() if sum(r["fit_id"] == value["fit_id"] for r in results["evaluations"].values()) > 1}
    save()
    render(results)
    results.setdefault("invocation_wall_s", []).append(time.perf_counter() - invocation_start)
    save()


def render(results):
    import csv
    from collections import Counter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for condition in CONDITIONS:
        for method in ("simple", "shallow", "depth_two", "transformer"):
            references = ["baseline"] if method in ("simple", "transformer") else (["cost_selected"] if condition == "zero" else ["frictionless_selected", "cost_selected"])
            for reference in references:
                values = [r for r in results["evaluations"].values() if (r["condition"], r["method"], r["reference"]) == (condition, method, reference)]
                row = dict(condition=condition, method=method, reference=reference, draws=len(values))
                for key in ("net_mse", "mean_error", "mean_turnover", "mean_notional_turnover", "mean_cost", "mean_proportional_cost", "mean_temporary_cost"):
                    row[key] = float(np.mean([r["metrics"][key] for r in values]))
                row["net_rmse"] = float(np.sqrt(row["net_mse"]))
                rows.append(row)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fit_times = {phase: sum(r["total_s"] for r in results["fits"].values() if r["phase"] == phase) for phase in ("tuning", "fresh", "baseline")}
    feature_s = sum(g["draw_s"] + sum(g["splits_s"].values()) for g in results["feature_groups"].values())
    evaluation_s = sum(e["evaluation_and_save_s"] for e in results["evaluations"].values())
    counts = Counter(r["optimizer_status"] for r in results["fits"].values())
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for method, label in (("simple", "Simple state hedge"), ("shallow", "Shallow tanh"), ("depth_two", "Depth-two tanh"), ("transformer", "Transformer")):
        series = [next(r for r in rows if r["condition"] == c and r["method"] == method and r["reference"] != "frictionless_selected")["net_rmse"] for c in CONDITIONS]
        ax.plot(range(3), series, marker="o", label=label)
    ax.set_xticks(range(3), ["Zero", "Proportional (5 bp)", "+ temporary (5 bp at 1 share)"])
    ax.set_ylabel("Test net RMSE (discounted price units)")
    ax.set_title("Fixed-premium Heston Asian · condition-selected policies")
    ax.legend()
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)
    lines = ["# Cost-aware Heston Asian reference", "", f"Completed {results['fit_count']} counted fits (cap 256), all 36 declared evaluations; {len(results['failures'])} failures.",
        f"Fixed premium: {PREMIUM:.12f}. Costs are illustrative and uncalibrated; see plan.json and README methods.", "",
        "| Condition | Method | Configuration source | Net RMSE (units) | Mean error (units) | Turnover (shares) | Costs (discounted units) |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['condition']} | {row['method']} | {row['reference']} | {row['net_rmse']:.6f} | {row['mean_error']:+.6f} | {row['mean_turnover']:.6f} | {row['mean_cost']:.6f} |")
    lines += ["", "Both reference configurations are retrained under the row's cost condition. Identical selected tuples reuse the same fits explicitly (see aliased_comparisons in results.json).", "", "Validation-selected reservoir reference configurations:"]
    for condition, choices in results["selected"].items():
        for method, value in choices.items():
            lines.append(f"- {condition}, {method}: tuple {value['candidate_index']} {value['candidate']}; validation net MSE {value['validation_mse']:.6f}.")
    lines += ["", f"Optimizer outcomes: {dict(counts)}. A step/epoch limit is not convergence or a failure. Norms characterize final iterates, while retained checkpoints minimize validation net MSE.",
        f"Measured wall time: data {results['data_generation_s']:.3f}s; feature groups {feature_s:.3f}s; fit totals by phase {fit_times}; evaluation and artifact saving {evaluation_s:.3f}s. Per-fit setup, optimization, validation and convergence-check times are saved in results.json; initial ridge solves and all restarts are counted.", "",
        "![Cost comparison](comparison.png)", "",
        "Conclusions concern these implemented policy classes under the fixed budget, not globally optimal hedging. Net-error training with costs is generally nonconvex, despite convex execution charges. One dataset partition, three reservoir draws, a sparse shortlist, limited restarts, uncalibrated coefficients and different readout coordinates limit comparisons. No position feedback, new information, price impact, filtering or new architecture was introduced.", "",
        "Run/resume from repository root:", "```bash",
        "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/workspace/.cache/rsmc-matplotlib XDG_CACHE_HOME=/workspace/.cache /workspace/.venvs/rsmc/bin/python scripts/cas_costs.py", "```"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:23]), flush=True)


if __name__ == "__main__":
    main()
