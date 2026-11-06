"""Protocol and runner integration checks; no research study is executed."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from rsmc.control import control_features
from rsmc.cost_hedging import hedge_readout_features
from rsmc.hedging import discounted_traded_increments, ewma_hedging_features
from rsmc.transfer_tasks import make_transfer_dataset


ROOT = Path(__file__).resolve().parents[1]


def load_runner(monkeypatch, root=ROOT):
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("cas_memory_under_test", root / "scripts/cas_memory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def all_seeds(plan):
    return [seed for ds in plan["datasets"]
            for seed in [ds["seed"], *[s for phase in ("tuning_seeds", "final_seeds")
                                      for draws in ds[phase].values() for s in draws]]]


def test_two_factor_budget_and_seeds_preserve_historical_protocols(monkeypatch):
    old = load_runner(monkeypatch)
    original_seeds = all_seeds(old.PLAN)
    assert old.PLAN["budget"]["total"] == 414
    old.configure_windows()
    historical_seeds = original_seeds + all_seeds(old.PLAN)
    assert old.PLAN["budget"]["total"] == 756

    runner = load_runner(monkeypatch)
    runner.configure_two_factor()
    plan = runner.PLAN
    assert plan["tasks"] == ["two_factor_heston"]
    assert plan["methods"] == ["static", "window5", "window20", "ewma", "reservoir"]
    assert plan["sizes"] == {"train": 4000, "val": 2000, "test": 8000}
    assert plan["n_steps"] == 50
    assert plan["ewma_rates"] == [.5, 5.]
    assert [(c["dim"], c["weight_scale"], c["bias_scale"], c["lam"])
            for c in plan["candidates"]] == [
                (64, s, 1., lam) for s in (.5, 1.) for lam in (0., .1, 10.)]
    assert len(plan["datasets"]) == 3
    selections = selection_fits = final_fits = 0
    for ds in plan["datasets"]:
        assert set(ds["tuning_seeds"]) == set(ds["final_seeds"]) == set(plan["methods"])
        for method in plan["methods"]:
            assert len(ds["tuning_seeds"][method]) == len(ds["final_seeds"][method]) == 3
            selections += 1
            selection_fits += len(plan["candidates"]) * len(ds["tuning_seeds"][method])
            final_fits += len(ds["final_seeds"][method])
    assert (selections, selection_fits, final_fits) == (15, 270, 45)
    assert plan["budget"] == dict(nonlinear_selection=270, affine_selection=0,
                                  nonlinear_final=45, affine_final=0, total=315)
    seeds = all_seeds(plan)
    assert len(seeds) == len(set(seeds)) == 93
    assert set(seeds).isdisjoint(historical_seeds)


def test_cli_modes_are_mutually_exclusive(monkeypatch):
    runner = load_runner(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["cas_memory.py", "--windows", "--two-factor"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


@pytest.mark.parametrize("method", ["static", "window5", "window20", "ewma", "reservoir"])
def test_runner_batches_preserve_nonlinear_inputs_and_shared_readout(monkeypatch, method):
    runner = load_runner(monkeypatch)
    runner.configure_two_factor()
    runner.PLAN["path_batch_size"] = 7
    data, times, rate, _, _ = make_transfer_dataset("two_factor_heston", {"train": 19}, 6, 701)
    sample = data["train"]
    model = runner.feature_map(method, runner.PLAN["candidates"][0], 702, 4)
    rows, feature_s, gain_s = runner.gain_rows(sample, times, rate, model, method)
    nonlinear = ewma_hedging_features(sample["features"], times) if method == "ewma" else None
    phi = hedge_readout_features(sample["features"], times, model, nonlinear_features=nonlinear)
    expected = control_features(phi, discounted_traded_increments(sample["spot"], times, rate))
    assert rows.shape == (19, 69)
    assert feature_s >= 0 and gain_s >= 0
    np.testing.assert_allclose(rows, expected, atol=1e-12, rtol=1e-12)


@pytest.fixture
def clean_runner_root(tmp_path):
    (tmp_path / "scripts").mkdir()
    for name in ("cas_memory.py", "cas_frictionless.py", "cas_transfer_report.py"):
        shutil.copy2(ROOT / "scripts" / name, tmp_path / "scripts" / name)
    shutil.copytree(ROOT / "rsmc", tmp_path / "rsmc", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def test_declaration_without_old_artifacts_and_source_guard(monkeypatch, clean_runner_root):
    root = clean_runner_root
    runner = load_runner(monkeypatch, root)
    monkeypatch.setattr(runner, "make_transfer_dataset", lambda *a, **kw: pytest.fail("Declaration generated paths"))
    monkeypatch.setattr(sys, "argv", ["cas_memory.py", "--two-factor", "--declare"])
    runner.main()
    out = root / "results/cas_memory/two_factor"
    plan = json.loads((out / "plan.json").read_text())
    declaration = json.loads((out / "declaration.json").read_text())
    assert declaration["fit_budget"]["total"] == 315
    assert declaration["unique_seeds"] == 93
    assert "rsmc/two_factor_heston.py" in declaration["source_sha256"]
    assert plan["models"]["two_factor_heston"]["kappa"] == [.5, 5.]
    assert not (out / "results.json").exists()
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    # Re-declaring unchanged sources is harmless; altered sources must be rejected
    # before simulation, including on the reporting-only path.
    load_runner(monkeypatch, root).main()
    changed = root / "rsmc/two_factor_heston.py"
    changed.write_bytes(changed.read_bytes() + b"\n# changed source\n")
    for flags in (["--declare"], [], ["--report-only"]):
        guarded = load_runner(monkeypatch, root)
        monkeypatch.setattr(guarded, "make_transfer_dataset", lambda *a, **kw: pytest.fail("Guard opened paths"))
        monkeypatch.setattr(sys, "argv", ["cas_memory.py", "--two-factor", *flags])
        with pytest.raises(RuntimeError, match="Sources differ"):
            guarded.main()
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not (out / "results.json").exists()


def test_resume_freezes_all_selections_before_opening_tests(monkeypatch, clean_runner_root):
    """Synthetic completed-fit records exercise bookkeeping, with no actual fits."""
    root = clean_runner_root
    runner = load_runner(monkeypatch, root)
    monkeypatch.setattr(sys, "argv", ["cas_memory.py", "--two-factor", "--declare"])
    runner.main()
    out = root / "results/cas_memory/two_factor"
    plan = json.loads((out / "plan.json").read_text())
    fits = {}
    for ds in plan["datasets"]:
        for method in plan["methods"]:
            for i, _ in enumerate(plan["candidates"]):
                for seed in ds["tuning_seeds"][method]:
                    key = f"{ds['id']}_tuning_{method}_c{i}_d{seed}"
                    fits[key] = dict(id=key, dataset=ds["id"], method=method,
                                     phase="tuning", candidate_index=i, seed=seed,
                                     status="completed", validation_mse=1.)
    state = dict(status="incomplete", fits=fits, selection={}, groups=[],
                 datasets={}, failures=[], invocations=[])
    (out / "results.json").write_text(json.dumps(state))
    resumed = load_runner(monkeypatch, root)

    class TestBoundaryReached(Exception):
        pass

    def fake_dataset(task, sizes, n_steps, seed, strike):
        if "test" in sizes:
            saved = json.loads((out / "results.json").read_text())
            assert sum(map(len, saved["selection"].values())) == 15
            assert saved["selection_sha256"] and saved["all_selections_frozen_utc"]
            assert all(choice["candidate_index"] == 0  # deterministic exact ties
                       for choices in saved["selection"].values() for choice in choices.values())
            raise TestBoundaryReached
        return {}, np.array([0., 1.]), .03, 0., {"model": {}, "dataset_spot_sha256": {}}

    monkeypatch.setattr(resumed, "make_transfer_dataset", fake_dataset)
    monkeypatch.setattr(resumed, "feature_map", lambda *a, **kw: pytest.fail("Rebuilt completed tuning features"))
    monkeypatch.setattr(resumed, "fit_fixed_premium_hedge", lambda *a, **kw: pytest.fail("Refitted completed records"))
    monkeypatch.setattr(sys, "argv", ["cas_memory.py", "--two-factor"])
    with pytest.raises(TestBoundaryReached):
        resumed.main()
    assert not list(out.glob("test_*.npz"))
