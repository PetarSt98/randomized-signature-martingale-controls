"""Auditable tables and three figures for the bounded CAS transfer study."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

TASK_LABELS = {"black_scholes": "Black–Scholes Asian", "heston": "Heston Asian",
               "basket": "Heston basket Asian"}
ROLE_LABELS = {"original": "Original setting", "reference": "Transferred reference",
               "task_selected": "Task selected", "simple": "Simple hedge",
               "transformer": "Transformer"}
CONDITION_LABELS = {"zero": "Frictionless", "combined": "Combined costs"}
COLORS = {"original": "#9467bd", "reference": "#1f77b4", "task_selected": "#ff7f0e",
          "simple": "#555555", "transformer": "#2ca02c"}
METRICS = ("net_mse", "mean_error", "mean_turnover", "mean_notional_turnover",
           "mean_proportional_cost", "mean_temporary_cost", "mean_cost")


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _configuration(index, plan):
    if index < len(plan["candidates"]):
        return plan["candidates"][index]
    return {"method": "simple" if index == 6 else "transformer"}


def incomplete_inventory(results, plan):
    """Name every missing fit and role obligation without inventing new runs."""
    expected = []
    for task in plan["tasks"]:
        for index in range(len(plan["candidates"])):
            for seed in plan["draw_seeds"]["tuning"]:
                base = f"{task}_tuning_-1_c{index}_d{seed}"
                expected += [base + "_ridge", base + "_combined_r0", base + "_combined_r1"]
        selected = results.get("selection", {}).get(task)
        if selected is None:
            continue
        condition_indices = {
            condition: {plan["original_index"], plan["reference_indices"][condition], selected[condition]["candidate_index"]}
            for condition in plan["conditions"]}
        indices = sorted(set.union(*condition_indices.values()))
        for split in range(3):
            for index in [*indices, 6]:
                for seed in plan["draw_seeds"]["final"] if index < 6 else [0]:
                    base = f"{task}_final_{split}_c{index}_d{seed}"
                    expected.append(base + "_ridge")
                    if index == 6 or index in condition_indices["combined"]:
                        expected += [base + "_combined_r0", base + "_combined_r1"]
            if task == "heston":
                for condition, seeds in plan["transformer_seeds"].items():
                    expected += [f"{task}_final_{split}_transformer_{condition}_{seed + 100 * split}" for seed in seeds]
    missing_roles = []
    for task in plan["tasks"]:
        for condition in plan["conditions"]:
            roles = ["original", "reference", "task_selected", "simple"] + (["transformer"] if task == "heston" else [])
            for role in roles:
                for split in range(3):
                    seeds = plan["draw_seeds"]["final"] if role in ("original", "reference", "task_selected") else [None]
                    for seed in seeds:
                        found = any(e["task"] == task and e["condition"] == condition and role in e["roles"]
                                    and e["split_index"] == split and (seed is None or e["seed"] == seed)
                                    for e in results["evaluations"].values())
                        if not found:
                            missing_roles.append(dict(task=task, condition=condition, role=role, split_index=split, draw_seed=seed))
    return {"missing_fit_ids": [key for key in expected if key not in results["fits"]],
            "unfinished_fit_ids": [key for key, row in results["fits"].items() if row["status"] != "completed"],
            "unselected_tasks": [task for task in plan["tasks"] if task not in results.get("selection", {})],
            "missing_comparisons": missing_roles}


def audit_saved_runs(results, plan, out):
    """Recompute every saved test metric and selection; fail on incomplete rows."""
    from rsmc.cost_hedging import cost_metrics

    out = Path(out)
    fits, evaluations = results["fits"], results["evaluations"]
    root = Path(__file__).resolve().parents[1]
    for relative, expected in results.get("source_sha256", {}).items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, f"Source changed: {relative}"
    if "reference_results_sha256" in results:
        assert hashlib.sha256((root / "results/cas_costs/results.json").read_bytes()).hexdigest() == results["reference_results_sha256"]
    incomplete = incomplete_inventory(results, plan)
    if any(incomplete.values()):
        (out / "missing_runs.json").write_text(json.dumps(incomplete, indent=2) + "\n")
        raise RuntimeError("Study incomplete; exact missing fit IDs and comparisons saved in missing_runs.json. Resume with the declared scripts/cas_transfer.py command.")
    assert len(fits) <= plan["budget"]["total_max"]
    assert not results["failures"]
    assert all(row["status"] == "completed" for row in fits.values())
    checked_histories = 0
    for row in fits.values():
        assert np.isfinite(row["validation_mse"])
        if row.get("history"):
            best = min(item["validation_mse"] for item in row["history"])
            np.testing.assert_allclose(row["validation_mse"], best, rtol=1e-12, atol=1e-12)
            checked_histories += 1
    for task in plan["tasks"]:
        for condition in plan["conditions"]:
            scores = []
            saved = results["selection"][task][condition]
            for index in range(len(plan["candidates"])):
                draw_scores = []
                for seed in plan["draw_seeds"]["tuning"]:
                    records = [row for row in fits.values() if row["phase"] == "tuning"
                               and row["task"] == task and row["condition"] == condition
                               and row["candidate_index"] == index and row["seed"] == seed]
                    assert len(records) == (1 if condition == "zero" else 2)
                    draw_scores.append(min(row["validation_mse"] for row in records))
                candidate = next(row for row in saved["candidates"] if row["candidate_index"] == index)
                np.testing.assert_allclose(candidate["draw_scores"], draw_scores, rtol=1e-12)
                score = float(np.mean(draw_scores))
                np.testing.assert_allclose(candidate["validation_mse"], score, rtol=1e-12)
                scores.append((score, index))
            score, index = min(scores)
            assert saved["candidate_index"] == index
            np.testing.assert_allclose(saved["validation_mse"], score, rtol=1e-12)
    seen = set()
    for key, row in evaluations.items():
        assert row["fit_id"] in fits
        fitted = fits[row["fit_id"]]
        assert fitted["condition"] == row["condition"]
        assert fitted["dataset_id"] == row["dataset_id"]
        assert fitted["premium"] == row["premium"]
        for role in row["roles"]:
            expected_index = (plan["original_index"] if role == "original" else
                              plan["reference_indices"][row["condition"]] if role == "reference" else
                              results["selection"][row["task"]][row["condition"]]["candidate_index"] if role == "task_selected" else
                              6 if role == "simple" else 7)
            assert row["candidate_index"] == expected_index, (key, role, expected_index)
        if row["method"] == "transformer":
            seeds = [seed + 100 * row["split_index"] for seed in plan["transformer_seeds"][row["condition"]]]
            alternatives = [r for r in fits.values() if r["method"] == "transformer" and r["condition"] == row["condition"] and r["dataset_id"] == row["dataset_id"]]
            assert sorted(r["seed"] for r in alternatives) == sorted(seeds)
            best = min(alternatives, key=lambda r: (r["validation_mse"], r["seed"]))
            expected_refits = {r["id"] for r in alternatives}
        elif row["condition"] == "combined":
            alternatives = [r for r in fits.values() if r["feature_group"] == row["feature_group"] and r["condition"] == "combined"]
            assert sorted(r["restart"] for r in alternatives) == [0, 1]
            best = min(alternatives, key=lambda r: (r["validation_mse"], r["restart"]))
            expected_refits = {row["feature_group"] + "_ridge", *[r["id"] for r in alternatives]}
        else:
            best = fits[row["feature_group"] + "_ridge"]
            expected_refits = {best["id"]}
        assert row["fit_id"] == best["id"]
        assert set(row["refit_ids"]) == expected_refits and len(row["refit_ids"]) == len(expected_refits)
        group = results["feature_groups"].get(row["feature_group"], {})
        training_s = sum(fits[fit_id]["total_s"] for fit_id in expected_refits) + group.get("draw_s", 0.) + sum(group.get("splits_s", {}).get(split, 0.) for split in ("train", "val"))
        np.testing.assert_allclose(row["training_s"], training_s, rtol=1e-12)
        np.testing.assert_allclose(row["deployment_s"], row["feature_generation_s"] + row["evaluation_s"], rtol=1e-12)
        assert row["premium"] == results["datasets"][row["dataset_id"]]["premium"]
        identity = (row["task"], row["dataset_id"], row["condition"], row["candidate_index"], row["seed"])
        assert identity not in seen, f"Duplicated configuration evaluation: {key}"
        seen.add(identity)
        with np.load(out / row["array_file"]) as saved:
            arrays = {name: saved[name] for name in saved.files}
        assert all(value.shape == (plan["sizes"]["test"],) for value in arrays.values())
        assert all(np.isfinite(value).all() for value in arrays.values())
        np.testing.assert_allclose(arrays["cost"], arrays["proportional_cost"] + arrays["temporary_cost"], rtol=1e-12, atol=1e-12)
        coefficients = plan["conditions"][row["condition"]]
        np.testing.assert_allclose(arrays["proportional_cost"], coefficients["kappa"] * arrays["notional_turnover"], rtol=1e-12, atol=1e-12)
        assert all(np.all(arrays[name] >= 0) for name in ("cost", "proportional_cost", "temporary_cost", "share_turnover", "notional_turnover"))
        if row["condition"] == "zero":
            assert all(np.count_nonzero(arrays[name]) == 0 for name in ("cost", "proportional_cost", "temporary_cost"))
        measured = cost_metrics(arrays["payoff"], row["premium"], arrays)
        np.testing.assert_allclose(arrays["errors"], measured.pop("errors"), rtol=1e-12, atol=1e-12)
        for metric, value in measured.items():
            np.testing.assert_allclose(row["metrics"][metric], value, rtol=1e-11, atol=1e-12)
        assert all(fit_id in fits for fit_id in row["refit_ids"])
    for task in plan["tasks"]:
        for condition in plan["conditions"]:
            roles = ["original", "reference", "task_selected", "simple"]
            if task == "heston":
                roles.append("transformer")
            for role in roles:
                rows = [row for row in evaluations.values() if row["task"] == task
                        and row["condition"] == condition and role in row["roles"]]
                assert len(rows) == (3 if role in ("simple", "transformer") else 9), (task, condition, role)
                assert {row["split_index"] for row in rows} == {0, 1, 2}
                for split in range(3):
                    group = [row for row in rows if row["split_index"] == split]
                    if role not in ("simple", "transformer"):
                        assert {row["seed"] for row in group} == set(plan["draw_seeds"]["final"])
    return {"fits_checked": len(fits), "evaluations_checked": len(evaluations),
            "checkpoint_histories_checked": checked_histories,
            "selection_conditions_checked": len(plan["tasks"]) * len(plan["conditions"]),
            "status": "passed"}


def aggregate(results, plan):
    """Pool draws within each dataset before computing descriptive variation."""
    rows, split_rows = [], []
    for task in plan["tasks"]:
        for condition in plan["conditions"]:
            roles = ["original", "reference", "task_selected", "simple"]
            if task == "heston":
                roles.append("transformer")
            for role in roles:
                values = [e for e in results["evaluations"].values()
                          if e["task"] == task and e["condition"] == condition and role in e["roles"]]
                by_split = []
                for split in range(3):
                    group = [e for e in values if e["split_index"] == split]
                    row = {"task": task, "condition": condition, "role": role,
                           "split_index": split, "dataset_id": group[0]["dataset_id"],
                           "dataset_seed": group[0]["dataset_seed"], "draws": len(group),
                           "premium": group[0]["premium"]}
                    row.update({key: float(np.mean([e["metrics"][key] for e in group])) for key in METRICS})
                    row["net_rmse"] = float(np.sqrt(row["net_mse"]))
                    for key in ("training_s", "deployment_s", "evaluation_s", "feature_generation_s"):
                        row[key] = float(np.mean([e[key] for e in group]))
                    row["final_compute_s"] = row["training_s"] + row["deployment_s"]
                    by_split.append(row)
                    split_rows.append(row)
                summary = {"task": task, "condition": condition, "role": role,
                           "datasets": 3, "evaluations": len(values),
                           "candidate_indices": ";".join(map(str, sorted({e["candidate_index"] for e in values})))}
                for key in (*METRICS, "training_s", "deployment_s", "evaluation_s", "feature_generation_s", "final_compute_s"):
                    summary[key] = float(np.mean([r[key] for r in by_split]))
                summary["net_rmse"] = float(np.sqrt(summary["net_mse"]))
                summary["dataset_rmse_sd"] = float(np.std([r["net_rmse"] for r in by_split], ddof=1))
                summary["dataset_compute_sd"] = float(np.std([r["final_compute_s"] for r in by_split], ddof=1))
                rows.append(summary)
    paired = []
    for task in plan["tasks"]:
        for condition in plan["conditions"]:
            deltas = []
            for split in range(3):
                group = [r for r in split_rows if r["task"] == task and r["condition"] == condition and r["split_index"] == split]
                reference = next(r for r in group if r["role"] == "reference")
                target = next(r for r in group if r["role"] == "task_selected")
                deltas.append(reference["net_rmse"] - target["net_rmse"])
            paired.append({"task": task, "condition": condition,
                           "reference_minus_target_rmse": float(np.mean(deltas)),
                           "dataset_difference_sd": float(np.std(deltas, ddof=1)),
                           **{f"split_{i}_difference": delta for i, delta in enumerate(deltas)}})
    return rows, split_rows, paired


def _per_run(results, plan):
    attached = defaultdict(list)
    for key, evaluation in results["evaluations"].items():
        attached[evaluation["fit_id"]].append((key, evaluation))
    rows = []
    for key, fitted in results["fits"].items():
        dataset = results["datasets"][fitted["dataset_id"]]
        feature = results["feature_groups"].get(fitted.get("feature_group"), {})
        row = {name: fitted.get(name, "") for name in (
            "id", "task", "dataset_id", "split_index", "phase", "condition", "method",
            "candidate_index", "seed", "restart", "premium", "feature_group", "status",
            "optimizer_status", "lam", "validation_mse", "total_s", "setup_s", "fit_s",
            "optimization_s", "validation_s", "gradient_inf", "best_step", "best_epoch", "failure")}
        row["dataset_seed"] = dataset["dataset_seed"]
        row["dependence_group"] = fitted["dataset_id"]
        row["configuration_json"] = json.dumps(fitted.get("config", fitted.get("configuration", _configuration(fitted["candidate_index"], plan))), sort_keys=True)
        row.update(plan["conditions"][fitted["condition"]])
        row.update({name: fitted.get(name, "") for name in ("rss_before_bytes", "rss_after_bytes", "process_peak_rss_bytes")})
        row.update({f"validation_{name}": value for name, value in fitted.get("validation_metrics", {}).items()})
        row["n_train"] = plan["sizes"]["train"]
        row["n_val"] = plan["sizes"]["val"]
        row["timings_json"] = json.dumps(fitted.get("timings", {}), sort_keys=True)
        row["feature_draw_s"] = feature.get("draw_s", "")
        row["feature_times_json"] = json.dumps(feature.get("splits_s", {}), sort_keys=True)
        row["feature_bytes_json"] = json.dumps(feature.get("feature_bytes", {}), sort_keys=True)
        row["feature_group_peak_rss_bytes"] = feature.get("process_peak_rss_bytes", "")
        linked = attached[key]
        assert len(linked) <= 1, "Identical role references must share one evaluation."
        row["evaluation_id"] = linked[0][0] if linked else ""
        row["roles"] = ";".join(linked[0][1]["roles"]) if linked else ""
        row["array_file"] = linked[0][1]["array_file"] if linked else ""
        row["refit_ids"] = ";".join(linked[0][1]["refit_ids"]) if linked else ""
        if linked:
            evaluation = linked[0][1]
            row.update({f"test_{name}": value for name, value in evaluation["metrics"].items()})
            row.update({name: evaluation[name] for name in ("evaluation_s", "save_s", "feature_generation_s", "deployment_s", "training_s")})
        rows.append(row)
    return rows


def render(results, plan, out):
    """Write tables, figures and an interpretation bounded by the declared study."""
    out = Path(out)
    started = time.perf_counter()
    tick = time.perf_counter()
    audit = audit_saved_runs(results, plan, out)
    timings = {"audit_s": time.perf_counter() - tick}
    tick = time.perf_counter()
    (out / "aggregation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    rows, splits, paired = aggregate(results, plan)
    _write_csv(out / "per_run.csv", _per_run(results, plan))
    _write_csv(out / "summary.csv", rows)
    _write_csv(out / "dataset_summary.csv", splits)
    _write_csv(out / "paired_transfer.csv", paired)
    timings["aggregation_tables_s"] = time.perf_counter() - tick
    tick = time.perf_counter()
    _figures(results, plan, rows, paired, out)
    timings["figures_s"] = time.perf_counter() - tick
    tick = time.perf_counter()
    _report(results, plan, rows, paired, audit, out)
    timings["report_s"] = time.perf_counter() - tick
    timings["total_s"] = time.perf_counter() - started
    (out / "reporting_timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    provenance = {"report_source": "scripts/cas_transfer_report.py",
                  "report_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "source_results_sha256": hashlib.sha256((out / "results.json").read_bytes()).hexdigest() if (out / "results.json").exists() else None}
    (out / "report_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {"summary": rows, "paired": paired, "audit": audit, "reporting_timings": timings}


def _figures(results, plan, rows, paired, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), layout="constrained")
    fig.suptitle("Declared configurations · S = shallow; D2 = depth-two; bias scale = 1", fontsize=12)
    for column, task in enumerate(plan["tasks"]):
        for line, condition in enumerate(plan["conditions"]):
            ax = axes[line, column]
            candidates = results["selection"][task][condition]["candidates"]
            x = [r["candidate_index"] for r in candidates]
            y = [r["validation_mse"] for r in candidates]
            sd = [np.std(r["draw_scores"], ddof=1) for r in candidates]
            ax.errorbar(x, y, yerr=sd, fmt="o", capsize=4, color="#1f77b4", label="Three-draw mean ± SD")
            winner = results["selection"][task][condition]
            ax.plot(winner["candidate_index"], winner["validation_mse"], "*", color="#d62728", markersize=13, label="Selected")
            labels = []
            for index in x:
                cfg = plan["candidates"][index]
                architecture = "S" if cfg["architecture"] == "shallow" else "D2-"
                labels.append(f"C{index}\n{architecture}{cfg['dim']}\ns={cfg['weight_scale']:g}\nλ={cfg['lam']:g}")
            ax.set_xticks(x, labels, fontsize=8)
            ax.set_title(f"{TASK_LABELS[task]}\n{CONDITION_LABELS[condition]}")
            ax.set_ylabel("Validation net MSE (discounted units²)")
            ax.set_xlabel("Declared complete configuration")
            ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(out / "validation.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), layout="constrained")
    markers = {"original": "s", "reference": "o", "task_selected": "*", "simple": "^", "transformer": "D"}
    for column, task in enumerate(plan["tasks"]):
        for line, condition in enumerate(plan["conditions"]):
            ax = axes[line, column]
            for row in rows:
                if (row["task"], row["condition"]) != (task, condition):
                    continue
                role = row["role"]
                ax.errorbar(row["final_compute_s"], row["net_rmse"], yerr=row["dataset_rmse_sd"],
                            fmt=markers[role], color=COLORS[role], markersize=9 if role == "task_selected" else 6,
                            markerfacecolor="none" if role == "reference" else COLORS[role],
                            capsize=4, label=ROLE_LABELS[role])
            ax.set_xscale("log")
            low, high = ax.get_xlim()
            multipliers = (1.,) if np.log10(high / low) >= 1.8 else (1., 2., 5.)
            ticks = [m * 10. ** exponent for exponent in range(int(np.floor(np.log10(low))) - 1, int(np.ceil(np.log10(high))) + 1)
                     for m in multipliers if low <= m * 10. ** exponent <= high]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("Final fitting + deployment (seconds, log scale)")
            ax.set_ylabel("Test net RMSE (discounted units)")
            ax.set_title(f"{TASK_LABELS[task]}\n{CONDITION_LABELS[condition]}")
            ax.grid(alpha=.2)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=9)
    fig.savefig(out / "error_cost.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    for offset, condition in zip((-.15, .15), plan["conditions"]):
        series = [next(row for row in paired if row["task"] == task and row["condition"] == condition) for task in plan["tasks"]]
        ax.errorbar(np.arange(3) + offset, [r["reference_minus_target_rmse"] for r in series],
                    yerr=[r["dataset_difference_sd"] for r in series], fmt="o", capsize=5,
                    label=CONDITION_LABELS[condition])
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(range(3), [TASK_LABELS[t] for t in plan["tasks"]])
    ax.set_ylabel("Transferred − task-selected net RMSE\n(discounted units; positive favors task selection)")
    ax.set_title("Matched final datasets: transferred versus task-specific settings")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.savefig(out / "transfer.png", dpi=160)
    plt.close(fig)


def _report(results, plan, rows, paired, audit, out):
    fits = results["fits"]
    counts = Counter(r.get("optimizer_status", "missing") for r in fits.values())
    fit_seconds = {phase: sum(r["total_s"] for r in fits.values() if r["phase"] == phase) for phase in ("tuning", "final")}
    feature_seconds = {phase: sum(g["draw_s"] + sum(g["splits_s"].values()) for g in results["feature_groups"].values() if g["phase"] == phase) for phase in ("tuning", "final")}
    evaluation_s = sum(e["evaluation_s"] for e in results["evaluations"].values())
    save_s = sum(e["save_s"] for e in results["evaluations"].values())
    peak = max([results.get("process_peak_rss_bytes", 0), *[g.get("process_peak_rss_bytes", 0) for g in results["feature_groups"].values()]])
    feature_peak = max(sum(g.get("feature_bytes", {}).values()) for g in results["feature_groups"].values())
    lines = ["# Final CAS transfer and reliability study", "",
             f"Completed **{len(fits)} counted fits** (cap {plan['budget']['total_max']}), **{len(results['evaluations'])} unique final evaluations**, and **{len(results['failures'])} failures**. Identical settings share evaluation records and list all roles explicitly. The saved-run aggregation audit passed.", "",
             "The study uses the existing Black–Scholes Asian, one-asset Heston Asian and three-asset Heston basket Asian tasks. Only frictionless and the validated combined-cost condition are run. All gains use discounted traded-price increments, with opening trades, rebalancing and terminal liquidation. The combined condition uses illustrative, uncalibrated kappa=eta=0.0005 and reference quantity 1 share per asset. Execution charges do not alter simulated prices.", "",
             "One independent selection partition per task selects settings, using six complete tuples and three validation reservoir draws. All task selections are frozen before generating any final dataset. Final evidence uses three fresh independent dataset partitions per task and three fresh reservoir draws per selected setting. Each partition has 4,000 training, 2,000 validation and 8,000 test paths, with 50 intervals. Each partition's float32-rounded discounted training payoff mean is frozen as its common premium across methods and costs. Readouts, preprocessing and dimension-compatible frozen matrices are refitted/redrawn on the target; only the tuple transfers.", "",
             "The original setting is the README quick-start tuple (shallow D=64, matrix/bias scales 1, lambda=0), adapted to this study's traded-gain and fixed-premium convention. The historical example was a pricing control; its fitted coefficients and historical performance are not hedge evidence. The transferred reference is the saved validation-selected depth-two D=32, scales 1, lambda=0.1 under the same cost condition. Transformer comparisons remain restricted to the predeclared Heston reference; its architecture is unchanged, with one frictionless and two combined-cost starts per final partition. The simple hedge is retained everywhere.", "",
             "## Declaration and selections", "", "| ID | Architecture | D | Matrix scale | Bias scale | Lambda |", "|---|---|---:|---:|---:|---:|"]
    for index, candidate in enumerate(plan["candidates"]):
        lines.append(f"| C{index} | {candidate['architecture']} | {candidate['dim']} | {candidate['weight_scale']} | {candidate['bias_scale']} | {candidate['lam']} |")
    lines += ["", "Selection minimizes unpenalized validation net MSE, first across checkpoints and the two declared positive-cost starts, then averaged over the three reservoir draws. The first tuple breaks exact ties. Zero-cost fits use fixed-premium ridge; positive-cost fits use the existing 120-step optimizer and two-start budget. All reservoir matrices remain frozen. The six-tuple list was not expanded after outcomes.", "", "| Task | Cost condition | Task-selected tuple | Validation net MSE (units²) |", "|---|---|---:|---:|"]
    for task in plan["tasks"]:
        for condition in plan["conditions"]:
            choice = results["selection"][task][condition]
            lines.append(f"| {TASK_LABELS[task]} | {CONDITION_LABELS[condition]} | C{choice['candidate_index']} | {choice['validation_mse']:.6f} |")
    lines += ["", "## Final results", "", "| Task | Condition | Setting | Net RMSE (units) | Dataset SD | Mean error (units) | Turnover (shares) | Costs (discounted units) |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {TASK_LABELS[row['task']]} | {CONDITION_LABELS[row['condition']]} | {ROLE_LABELS[row['role']]} | {row['net_rmse']:.6f} | {row['dataset_rmse_sd']:.6f} | {row['mean_error']:+.6f} | {row['mean_turnover']:.6f} | {row['mean_cost']:.6f} |")
    lines += ["", "RMSE is the square root of mean unpenalized MSE across the three draws and three datasets. Errors are never recentered. The three draws within a dataset share its training, validation and test paths, premium and deterministic preprocessing. The same three final reservoir seeds are reused across final datasets, so dataset error bars describe dataset variation conditional on this fixed three-draw ensemble, not independent resampling of all sources of randomness. Methods and cost conditions are also matched on these paths. Consequently, nine reservoir evaluations are not nine independent dataset replicates. Dataset SD first pools the three draw MSEs within each dataset, takes its RMSE, then reports the sample SD across the three dataset RMSEs. Transformer seeds change by the declared +100 offset between final partitions, so its dataset SD also includes the declared neural initialization/training variation; the fixed-ensemble interpretation applies to reservoir rows. These are descriptive error bars, not confidence intervals; three datasets give limited reliability evidence.", "",
             "| Task | Condition | Paired reference − task RMSE (units) | Dataset difference SD |", "|---|---|---:|---:|"]
    for row in paired:
        lines.append(f"| {TASK_LABELS[row['task']]} | {CONDITION_LABELS[row['condition']]} | {row['reference_minus_target_rmse']:+.6f} | {row['dataset_difference_sd']:.6f} |")
    findings = []
    selected_indices = {results["selection"][task][condition]["candidate_index"] for task in plan["tasks"] for condition in plan["conditions"]}
    if selected_indices == {plan["original_index"]}:
        findings.append("Every task and cost condition selected C0, the documented original setting. Original and task-selected rows therefore alias the same retrained readouts and final evaluations; they are not separate replications. In this shortlist, task selection retained the original hyperparameters rather than the transferred reference.")
    better_cells = 0
    for task in plan["tasks"]:
        for condition in plan["conditions"]:
            target = next(r for r in rows if (r["task"], r["condition"], r["role"]) == (task, condition, "task_selected"))
            reference = next(r for r in rows if (r["task"], r["condition"], r["role"]) == (task, condition, "reference"))
            better_cells += target["net_rmse"] < reference["net_rmse"]
    findings.append(f"The task-selected setting had lower pooled final test RMSE than the transferred reference in {better_cells} of {len(plan['tasks']) * len(plan['conditions'])} task/condition cells. This is a configuration-level observation on these datasets: architecture, width and regularization differ jointly, so it does not isolate an architecture advantage or establish that target tuning universally improves hedging.")
    tf_rows = [r for r in rows if r["role"] == "transformer"]
    if tf_rows and all(r["net_rmse"] < min(other["net_rmse"] for other in rows if other["task"] == "heston" and other["condition"] == r["condition"] and other["role"] != "transformer") for r in tf_rows):
        targets = [r for r in rows if r["task"] == "heston" and r["role"] == "task_selected"]
        tf_low, tf_high = min(r["training_s"] for r in tf_rows), max(r["training_s"] for r in tf_rows)
        rs_low, rs_high = min(r["training_s"] for r in targets), max(r["training_s"] for r in targets)
        cost_phrase = " with greater final fitting expense" if tf_low > rs_high else ""
        findings.append(f"On the retained Heston reference comparisons, the Transformer achieved the lowest pooled test RMSE{cost_phrase}. Mean standalone training and validation-selection time was {tf_low:.2f}–{tf_high:.2f} s across the two cost conditions, compared with {rs_low:.2f}–{rs_high:.2f} s for the task-selected reservoir. The combined-cost figures include both declared starts; all timing comparisons remain descriptive.")
    lines += ["", "Positive paired differences favor task selection. Each difference uses matched final datasets after pooling their reservoir draws. Exact zeros can reflect deduplicated identical settings. Test outcomes never revise a selected setting.", "", "## Interpretation", ""]
    for finding in findings:
        lines += [finding, ""]
    lines += [
             "## Computation, caching and memory", "",
             f"Actual unique-fit wall time: tuning {fit_seconds['tuning']:.3f} s; final fitting {fit_seconds['final']:.3f} s. Feature-generation wall time: tuning {feature_seconds['tuning']:.3f} s; final {feature_seconds['final']:.3f} s. Final cash-flow/inference evaluation {evaluation_s:.3f} s; array saving {save_s:.3f} s. Data generation totals {sum(d.get('generation_s', 0) for d in results['datasets'].values()):.3f} s. Invocation wall times are recorded separately in results.json, including orchestration and checkpoint writing but excluding subsequent audit/render. reporting_timings.json records the report audit, aggregation, table and figure rendering times separately.", "",
             "Per-run training_s is the standalone cost of drawing that frozen reservoir, generating training/validation features, and all readout fits required for its condition, including ridge initialization and both positive-cost starts. Deployment_s is test feature generation plus position inference and cash-flow evaluation. These standalone comparison timings can reuse a shared initializer or cached features across conditions; do not sum them to infer actual total runtime. The totals above count each fit and feature group once. Tuning expense remains separate from final fitting and deployment. Timing includes validation checkpoint selection and neural checkpoint serialization; final NPZ artifact saving is reported separately. Hardware, runtime variation and differing numbers of optimizer starts limit timing comparisons.", "",
             "Training/validation reservoir features are cached in memory for one configuration and draw, shared by frictionless and combined-cost readout fits and both restarts, then discarded. No persistent feature cache is used. Test features are generated only for retained comparisons. Completed fit/evaluation records are reused on resume; deterministic data and feature regeneration may incur extra costs recorded by the invocation. Per-run feature-group identifiers expose the sharing, and feature times repeated in per_run.csv must not be added repeatedly.", "",
             f"Measured process high-water resident memory: {peak / 2**20:.2f} MiB. Largest recorded feature group (sum of cached feature-array nbytes): {feature_peak / 2**20:.2f} MiB. Process RSS comes from /proc and cumulative ru_maxrss; it includes interpreter, libraries, optimizer buffers and allocator retention and is not an isolated per-model peak. Array nbytes measures the named feature arrays only and excludes temporary arrays. Measurements are descriptive of this execution environment.", "",
             f"Optimizer outcomes: {dict(counts)}. A budget-exhausted status is not numerical convergence; selecting a checkpoint does not prove stationarity. Recorded failure fields remain distinct from these optimizer-limit outcomes.", "",
             "## Figures", "", "![Validation across declared tuples](validation.png)", "", "Figure 1. Mean unpenalized validation net MSE with ± one sample SD across three reservoir draws on the task's single selection dataset. S denotes shallow, D2 depth-two, s the matrix scale and λ the readout penalty; bias scale is 1 throughout. Stars mark the frozen choices; bars describe reservoir randomness conditional on that dataset, not dataset uncertainty.", "",
             "![Final error versus computational cost](error_cost.png)", "", "Figure 2. Pooled final net RMSE versus mean standalone final fitting plus deployment wall time, excluding tuning; the horizontal axis is logarithmic seconds. Vertical bars show ± one SD across three dataset-level RMSEs. Exact aliases overlap. No hardware-normalized or asymptotic speed claim is made.", "",
             "![Paired transferred versus task-specific settings](transfer.png)", "", "Figure 3. Mean matched-dataset reference-minus-task-selected RMSE, with ± one sample SD of the three dataset differences. Positive values favor task selection; identical transferred and target tuples are explicitly shared.", "",
             "## Reproducibility and limits", "",
             "per_run.csv is the consistent table of every counted fit, including all tuning candidates, initializers, optimizer restarts, configuration JSON, dataset/reservoir or neural seeds, premium, convergence status, failures, timing and linked final errors/turnover/costs. Unselected starts have blank test columns because they were not evaluated. results.json retains checkpoints, selection scores, coefficients, hashes and memory records; per-evaluation NPZ files retain payoff, error and cash flows. summary.csv, dataset_summary.csv and paired_transfer.csv contain the derived aggregates. aggregation_audit.json records independent recomputation of every saved metric, validation selection and expected role/dataset/draw counts.", "",
             "Conclusions concern the implemented policy classes and these fixed budgets, not globally optimal hedging. The shortlist is sparse; architecture, width and regularization vary jointly between configurations (including C0 versus C1), so these comparisons do not isolate an architecture effect. Positive-cost objectives are generally nonconvex, execution coefficients are uncalibrated, and only three independent final dataset partitions are available. Target selection uses one independent partition per task, so selection uncertainty itself is not repeated across final datasets. Conditional reservoir-draw variation and dataset variation remain distinct. The Heston simulator retains its existing discretization. Historical filtering experiments and separate reconstruction mathematics remain outside this study.", "",
             "Run/resume from the repository root:", "```bash", "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.cache/rsmc-matplotlib XDG_CACHE_HOME=/workspace/.cache /workspace/.venvs/rsmc/bin/python scripts/cas_transfer.py", "```", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
