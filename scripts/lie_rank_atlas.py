"""Command-line entry point for the homogeneous free-Lie rank atlas."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _find_repo_root() -> Path:
    start = Path(__file__).resolve().parent
    for folder in (start, *start.parents):
        if (folder / "lie_reservoir").is_dir():
            return folder
    raise RuntimeError(
        "could not locate a repository root containing lie_reservoir"
    )


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lie_reservoir.rank_atlas import (  # noqa: E402
    ARCHITECTURES,
    DEFAULT_MAX_MATRIX_ENTRIES,
    DEFAULT_POINT_SCALE,
    DEFAULT_POINT_SEED,
    DEFAULT_RELATIVE_TOLERANCES,
    DEFAULT_ROW_OVERSAMPLING,
    DEFAULT_WEIGHT_SEED,
    format_rank_table,
    full_atlas_cases,
    make_rank_atlas_case,
    run_rank_atlas,
    smoke_atlas_cases,
    write_rank_atlas,
)


def _parse_relative_tolerances(value: str) -> tuple[float, ...]:
    try:
        tolerances = tuple(
            float(part.strip())
            for part in value.split(",")
            if part.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "relative tolerances must be comma-separated real numbers"
        ) from exc
    if not tolerances:
        raise argparse.ArgumentTypeError(
            "at least one relative tolerance is required"
        )
    return tolerances


def _add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: Path,
) -> None:
    parser.add_argument(
        "--weight-seed",
        type=int,
        default=DEFAULT_WEIGHT_SEED,
    )
    parser.add_argument(
        "--point-seed",
        type=int,
        default=DEFAULT_POINT_SEED,
    )
    parser.add_argument(
        "--row-oversampling",
        type=int,
        default=DEFAULT_ROW_OVERSAMPLING,
        help="Requested extra scalar rows beyond the Witt dimension.",
    )
    parser.add_argument(
        "--point-scale",
        type=float,
        default=DEFAULT_POINT_SCALE,
    )
    parser.add_argument(
        "--relative-tolerances",
        type=_parse_relative_tolerances,
        default=DEFAULT_RELATIVE_TOLERANCES,
        metavar="RTOL1,RTOL2,...",
    )
    parser.add_argument(
        "--max-matrix-entries",
        type=int,
        default=DEFAULT_MAX_MATRIX_ENTRIES,
        help="Per-case allocation cap checked before sampling or autodiff.",
    )
    parser.add_argument(
        "--shallow-weight-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--shallow-bias-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--outer-weight-scale",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--diagonal-scale",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute numerical ranks separately for homogeneous "
            "standard-Lyndon bases."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run six small cases: three architectures and degrees 1,2.",
    )
    _add_common_arguments(
        smoke_parser,
        default_output_dir=REPO_ROOT / "data/lie_rank_atlas/smoke",
    )

    single_parser = subparsers.add_parser(
        "single",
        help="Run one architecture, dimension, and homogeneous degree.",
    )
    single_parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        required=True,
    )
    single_parser.add_argument("--input-dim", type=int, required=True)
    single_parser.add_argument("--state-dim", type=int, required=True)
    single_parser.add_argument("--degree", type=int, required=True)
    _add_common_arguments(
        single_parser,
        default_output_dir=REPO_ROOT / "data/lie_rank_atlas/single",
    )

    full_parser = subparsers.add_parser(
        "full",
        help="Run the unexecuted-by-default 120-case reconnaissance grid.",
    )
    _add_common_arguments(
        full_parser,
        default_output_dir=REPO_ROOT / "data/lie_rank_atlas/full",
    )
    return parser


def _case_options(args: argparse.Namespace) -> dict[str, object]:
    return {
        "weight_seed": args.weight_seed,
        "point_seed": args.point_seed,
        "row_oversampling": args.row_oversampling,
        "point_scale": args.point_scale,
        "relative_tolerances": args.relative_tolerances,
        "shallow_weight_scale": args.shallow_weight_scale,
        "shallow_bias_scale": args.shallow_bias_scale,
        "outer_weight_scale": args.outer_weight_scale,
        "diagonal_scale": args.diagonal_scale,
    }


def main(argv: list[str] | None = None) -> int:
    """Run one CLI subcommand and persist its complete numerical evidence."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    case_options = _case_options(args)

    if args.command == "smoke":
        cases = smoke_atlas_cases(**case_options)
        stem = "smoke_rank_atlas"
    elif args.command == "single":
        cases = (
            make_rank_atlas_case(
                architecture=args.architecture,
                input_dim=args.input_dim,
                state_dim=args.state_dim,
                degree=args.degree,
                **case_options,
            ),
        )
        stem = "single_rank_atlas"
    else:
        cases = full_atlas_cases(**case_options)
        stem = "full_rank_atlas"

    started = time.perf_counter()
    results = run_rank_atlas(
        cases,
        max_matrix_entries=args.max_matrix_entries,
    )
    csv_path, json_path = write_rank_atlas(
        results,
        args.output_dir,
        stem=stem,
    )
    elapsed_seconds = time.perf_counter() - started
    completed_count = sum(result.status == "completed" for result in results)
    skipped_count = len(results) - completed_count

    print(format_rank_table(results))
    print()
    print(
        f"cases={len(results)} completed={completed_count} "
        f"skipped={skipped_count} elapsed_seconds={elapsed_seconds:.3f}"
    )
    print(f"csv={csv_path.resolve()}")
    print(f"json={json_path.resolve()}")
    print("Interpretation: numerical homogeneous-rank evidence only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
