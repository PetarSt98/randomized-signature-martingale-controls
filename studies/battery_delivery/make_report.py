"""Regenerate the study’s results table and comparison figure."""

import csv
import json
from pathlib import Path

import numpy as np

from run_pilot import PLAN, unit_call


METRICS = ("volume_mse", "quote_mse", "hedge_mse")


def observer_scores(primary, observer):
    """Score the model-aware observer under the same forward prices and premium."""
    with np.load(primary) as test, np.load(observer) as estimates:
        volume = test["volume"]
        np.testing.assert_array_equal(volume, estimates["volume"])
        error = estimates["estimated"] - volume
        price, delta, residual = unit_call(test["forward"])
        premium_error = (volume - test["mean_train_volume"]) * price
        hedge_error = PLAN["settlement_innovation_sd"] * delta * error
        return {
            "volume_mse": np.mean(error**2),
            "quote_mse": np.mean((error * price)**2),
            "hedge_mse": np.mean(volume**2 * residual + premium_error**2 + hedge_error**2),
        }


def summarize(results):
    by_method = {}
    for split in (1, 2, 3):
        folder = f"split_{split}"
        scores = json.loads((results / folder / "scores.json").read_text())["scores"]
        scores.update(json.loads((results / "extra" / folder / "scores.json").read_text()))
        scores["calibrated_equations"] = observer_scores(
            results / folder / "predictions.npz", results / "observer" / f"{folder}.npz"
        )
        confirmation = json.loads((results / "confirmation" / folder / "scores.json").read_text())
        for sample, models in (("main", scores), ("confirmation", confirmation)):
            for method, model in models.items():
                by_method.setdefault((sample, method), []).append([model[k] for k in METRICS])

    with (results / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("sample", "method", "split", "delivery_rmse_mwh", "quote_rmse_eur", "hedge_rmse_eur"))
        for (sample, method), splits in sorted(by_method.items()):
            # Splits within each study contain equal numbers of test histories.
            pooled = np.mean(splits, axis=0)
            for label, losses in zip(("1", "2", "3", "pooled"), [*splits, pooled]):
                writer.writerow([sample, method, label, *(f"{np.sqrt(x):.7f}" for x in losses)])


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = {
        "static": "Latest readings",
        "ewma": "Eight running averages",
        "raw_history": "Full history (static tanh)",
        "reservoir": "CAS reservoir",
        "generic_moments": "Joint sensor statistics",
        "circuit": "Battery pulse features",
        "calibrated_equations": "Calibrated battery equations",
    }
    with (results / "metrics.csv").open() as stream:
        pooled = {row["method"]: row for row in csv.DictReader(stream)
                  if row["sample"] == "main" and row["split"] == "pooled"}
    colors = ["#116c72" if m == "reservoir" else "#8496a5" for m in methods]
    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "battery-delivery"}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
        for ax, metric, title in zip(axes, ("quote_rmse_eur", "hedge_rmse_eur"),
                                    ("Option quote", "Hedged settlement")):
            values = [float(pooled[m][metric]) for m in methods]
            bars = ax.barh(list(methods.values()), values, color=colors, height=0.55)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
            ax.set(xlim=(0, max(values) * 1.17), xlabel="RMSE (€ per contract)")
            ax.set_title(title, loc="left")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
        axes[0].invert_yaxis()
        fig.suptitle("15-minute battery delivery option · 30,000 simulated test histories")
        figure = results / "comparison.svg"
        fig.savefig(figure, metadata={"Date": None})
        plt.close(fig)
        # Drop the path-command line padding added by the SVG backend.
        lines = figure.read_text(encoding="utf-8").splitlines()
        figure.write_bytes(("\n".join(line.rstrip() for line in lines) + "\n").encode("utf-8"))


if __name__ == "__main__":
    results = Path(__file__).resolve().parent / "results"
    summarize(results)
    plot(results)
