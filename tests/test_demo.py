"""Smoke test for the four-row benchmark demo orchestration."""

from __future__ import annotations

import importlib.util
import sys
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
    rows = demo.run_demo(
        sizes={"train": 80, "val": 40, "test": 60},
        n_steps=4,
        reservoir_dims=(4,),
        signature_levels=(2,),
        lambdas=(0.0, 1.0),
        signature_caps={},
        print_output=False,
    )
    assert len(rows) == 8
    assert sorted({row["d"] for row in rows}) == [3, 4, 8, 12]
    assert sorted({row["m"] for row in rows}) == [1, 2, 6, 9]
    for d in (3, 4, 8, 12):
        case_rows = [row for row in rows if row["d"] == d]
        assert {row["method"] for row in case_rows} == {"reservoir", "signature"}


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
