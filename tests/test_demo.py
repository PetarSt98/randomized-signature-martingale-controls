"""Smoke test for the four-row benchmark demo orchestration."""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_demo_module():
    path = ROOT / "scripts" / "demo.py"
    spec = importlib.util.spec_from_file_location("demo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_demo_smoke_reports_four_real_dimensions() -> None:
    demo = _load_demo_module()
    assert max(demo.RIDGE_LAMBDAS) >= 100_000.0
    rows = demo.run_demo(
        sizes={"train": 80, "val": 40, "test": 60},
        n_steps=4,
        reservoir_dims=(4,),
        signature_levels=(2,),
        lambdas=(0.0, 1.0),
        print_output=False,
    )
    assert len(rows) == 8
    assert sorted({row["d"] for row in rows}) == [3, 4, 8, 12]
    assert sorted({row["m"] for row in rows}) == [1, 2, 6, 9]
    assert {row["n_train"] for row in rows} == {80}
    assert {row["n_val"] for row in rows} == {40}
    assert {row["n_eval"] for row in rows} == {60}
    for d in (3, 4, 8, 12):
        case_rows = [row for row in rows if row["d"] == d]
        assert {row["method"] for row in case_rows} == {"reservoir", "signature"}


def test_expensive_signature_levels_are_disabled_by_default() -> None:
    demo = _load_demo_module()
    rows = demo.run_demo(
        sizes={"train": 40, "val": 20, "test": 30},
        n_steps=3,
        reservoir_dims=(4,),
        signature_levels=(4,),
        lambdas=(1.0,),
        print_output=False,
    )
    assert len(rows) == 6
    assert {(row["d"], row["label"]) for row in rows if row["method"] == "signature"} == {
        (3, "signature L=4"),
        (4, "signature L=4"),
    }


def test_final_summary_uses_same_case_rows() -> None:
    demo = _load_demo_module()
    rows = [
        {
            "model": "case",
            "product": "product",
            "d": 12,
            "m": 9,
            "method": "reservoir",
            "label": "reservoir D=16",
            "variance_reduction": 2.0,
            "feature_test_s": 0.1,
        },
        {
            "model": "case",
            "product": "product",
            "d": 12,
            "m": 9,
            "method": "signature",
            "label": "signature L=2",
            "variance_reduction": 1.0,
            "feature_test_s": 0.2,
        },
        {
            "model": "case",
            "product": "product",
            "d": 12,
            "m": 9,
            "method": "signature",
            "label": "signature L=4",
            "variance_reduction": 99.0,
            "feature_test_s": 0.3,
        },
    ]
    out = io.StringIO()
    with redirect_stdout(out):
        demo.print_final_summary(rows)
    text = out.getvalue()
    assert "signature L=4" in text
    assert "All rows use the same train/val/test size" in text


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
