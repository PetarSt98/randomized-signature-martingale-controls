"""Execute and aggregate the targeted homogeneous-rank replication study.

Stage A is produced by ``scripts/lie_rank_atlas.py full``. This script reads
that result, runs the predeclared Stage B replication grid, and writes the
combined research artifacts. It does not change the rank calculation.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.linalg
import torch


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

from lie_reservoir import (  # noqa: E402
    RankAtlasCase,
    RankAtlasResult,
    build_homogeneous_evaluation_matrix,
    compute_rank_diagnostics,
    format_lie_node,
    lyndon_basis,
    make_rank_atlas_case,
    sample_evaluation_points,
    sample_parameters,
    witt_dimension,
    write_rank_atlas,
)
from lie_reservoir.rank_atlas import (  # noqa: E402
    DEFAULT_ROW_OVERSAMPLING,
)


OUTPUT_DIR = REPO_ROOT / "research_output"
STAGE_A_JSON = OUTPUT_DIR / "stage_a" / "full_rank_atlas.json"
STAGE_A_CSV = OUTPUT_DIR / "stage_a" / "full_rank_atlas.csv"
STAGE_B_DIR = OUTPUT_DIR / "stage_b"
SELECTED_TARGET_DIR = OUTPUT_DIR / "selected_target"

WEIGHT_SEEDS = (1729, 1730, 1731, 1732, 1733)
POINT_SEEDS = (2718, 2719, 2720)
RELATIVE_TOLERANCES = (1e-8, 1e-10, 1e-12)


@dataclass(frozen=True)
class ReplicationSpec:
    architecture: str
    input_dim: int
    state_dim: int
    degrees: tuple[int, ...]


REPLICATION_SPECS = (
    ReplicationSpec(
        architecture="shallow_biased_tanh",
        input_dim=2,
        state_dim=1,
        degrees=(1, 2, 3, 4),
    ),
    ReplicationSpec(
        architecture="shallow_bias_free_tanh",
        input_dim=2,
        state_dim=1,
        degrees=(4,),
    ),
    ReplicationSpec(
        architecture="depth_two_exponential",
        input_dim=2,
        state_dim=1,
        degrees=(4,),
    ),
    ReplicationSpec(
        architecture="shallow_biased_tanh",
        input_dim=3,
        state_dim=1,
        degrees=(3, 4),
    ),
    ReplicationSpec(
        architecture="shallow_bias_free_tanh",
        input_dim=3,
        state_dim=1,
        degrees=(3, 4),
    ),
    ReplicationSpec(
        architecture="depth_two_exponential",
        input_dim=3,
        state_dim=1,
        degrees=(3, 4),
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as json_file:
        json.dump(
            payload,
            json_file,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        json_file.write("\n")


def _group_stem(spec: ReplicationSpec, degree: int) -> str:
    return (
        f"{spec.architecture}_input{spec.input_dim}_"
        f"state{spec.state_dim}_degree{degree}"
    )


def _write_group_integrity_manifest(
    stem: str,
    json_path: Path,
    csv_path: Path,
    matrix_path: Path,
    manifest_path: Path,
) -> None:
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "group_stem": stem,
            "artifacts": {
                "json": {
                    "name": json_path.name,
                    "sha256": _sha256(json_path),
                },
                "csv": {
                    "name": csv_path.name,
                    "sha256": _sha256(csv_path),
                },
                "matrices": {
                    "name": matrix_path.name,
                    "sha256": _sha256(matrix_path),
                },
            },
        },
    )


def _validate_group_integrity_manifest(
    stem: str,
    json_path: Path,
    csv_path: Path,
    matrix_path: Path,
    manifest_path: Path,
) -> None:
    manifest = _load_json(manifest_path)
    expected_artifacts = {
        "json": json_path,
        "csv": csv_path,
        "matrices": matrix_path,
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("group_stem") != stem
        or set(manifest.get("artifacts", {})) != set(expected_artifacts)
    ):
        raise RuntimeError(
            f"invalid Stage B integrity manifest: {manifest_path}"
        )
    for label, path in expected_artifacts.items():
        record = manifest["artifacts"][label]
        if (
            record.get("name") != path.name
            or record.get("sha256") != _sha256(path)
        ):
            raise RuntimeError(
                f"Stage B artifact failed integrity validation: {path}"
            )


def _run_id(stage: str, metadata: dict[str, Any]) -> str:
    return (
        f"{stage}_{metadata['architecture']}_"
        f"input{metadata['input_dim']}_state{metadata['state_dim']}_"
        f"degree{metadata['degree']}_weight{metadata['weight_seed']}_"
        f"point{metadata['point_seed']}"
    )


def _expanded_row_oversampling(
    input_dim: int,
    state_dim: int,
    degree: int,
) -> int:
    column_count = witt_dimension(input_dim, degree)
    stage_a_point_count = math.ceil(
        (column_count + DEFAULT_ROW_OVERSAMPLING) / state_dim
    )
    expanded_point_count = 2 * stage_a_point_count
    return expanded_point_count * state_dim - column_count


def _extension_point_seed(point_seed: int) -> int:
    return 1_000_000_000 + point_seed


def _sample_expanded_points(case: RankAtlasCase) -> torch.Tensor:
    baseline_case = RankAtlasCase(
        vector_field_config=case.vector_field_config,
        degree=case.degree,
        weight_seed=case.weight_seed,
        point_seed=case.point_seed,
        row_oversampling=DEFAULT_ROW_OVERSAMPLING,
        point_scale=case.point_scale,
        relative_tolerances=case.relative_tolerances,
    )
    baseline_points = sample_evaluation_points(baseline_case)
    additional_point_count = (
        case.point_count - baseline_case.point_count
    )
    if additional_point_count <= 0:
        raise RuntimeError(
            "Stage B must add points beyond the Stage A baseline."
        )
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(_extension_point_seed(case.point_seed))
    additional_points = torch.randn(
        (additional_point_count, case.state_dim),
        generator=random_generator,
        dtype=torch.float64,
        device="cpu",
    ) * float(case.point_scale)
    return torch.cat((baseline_points, additional_points), dim=0)


def _run_replication_case(case: RankAtlasCase) -> RankAtlasResult:
    started = time.perf_counter()
    basis = lyndon_basis(case.input_dim, case.degree)
    parameters = sample_parameters(
        case.vector_field_config,
        seed=case.weight_seed,
    )
    evaluation_points = _sample_expanded_points(case)
    matrix = build_homogeneous_evaluation_matrix(
        config=case.vector_field_config,
        parameters=parameters,
        degree=case.degree,
        evaluation_points=evaluation_points,
    )
    diagnostics = compute_rank_diagnostics(
        matrix,
        relative_tolerances=case.relative_tolerances,
    )
    return RankAtlasResult(
        case=case,
        status="completed",
        basis_labels=tuple(format_lie_node(node) for node in basis),
        matrix_entry_count=case.row_count * case.column_count,
        max_matrix_entries=10_000,
        elapsed_seconds=time.perf_counter() - started,
        parameters=parameters,
        evaluation_points=evaluation_points,
        evaluation_matrix=matrix,
        diagnostics=diagnostics,
        skip_reason=None,
    )


def _validate_completed_group(
    spec: ReplicationSpec,
    degree: int,
    json_path: Path,
    csv_path: Path,
    matrix_path: Path,
    manifest_path: Path,
) -> None:
    stem = _group_stem(spec, degree)
    _validate_group_integrity_manifest(
        stem,
        json_path,
        csv_path,
        matrix_path,
        manifest_path,
    )
    payload = _load_json(json_path)
    results = payload.get("results")
    expected_case_count = len(WEIGHT_SEEDS) * len(POINT_SEEDS)
    if (
        not isinstance(results, list)
        or len(results) != expected_case_count
        or payload.get("completed_count") != expected_case_count
        or payload.get("skipped_count") != 0
    ):
        raise RuntimeError(f"invalid completed group payload: {json_path}")

    expected_pairs = {
        (weight_seed, point_seed)
        for weight_seed in WEIGHT_SEEDS
        for point_seed in POINT_SEEDS
    }
    expected_row_oversampling = _expanded_row_oversampling(
        spec.input_dim,
        spec.state_dim,
        degree,
    )
    seen_pairs: set[tuple[int, int]] = set()
    expected_run_ids: set[str] = set()
    expected_shapes: dict[str, tuple[int, int]] = {}
    for result in results:
        metadata = result.get("metadata", {})
        pair = (
            metadata.get("weight_seed"),
            metadata.get("point_seed"),
        )
        if (
            result.get("status") != "completed"
            or metadata.get("architecture") != spec.architecture
            or metadata.get("input_dim") != spec.input_dim
            or metadata.get("state_dim") != spec.state_dim
            or metadata.get("degree") != degree
            or metadata.get("row_oversampling")
            != expected_row_oversampling
            or tuple(metadata.get("relative_tolerances", ()))
            != RELATIVE_TOLERANCES
            or pair not in expected_pairs
            or pair in seen_pairs
        ):
            raise RuntimeError(
                f"group result does not match the protocol: {json_path}"
            )
        seen_pairs.add(pair)
        run_id = _run_id("stage_b", metadata)
        expected_run_ids.add(run_id)
        expected_shapes[run_id] = (
            metadata["row_count"],
            metadata["column_count"],
        )
    if seen_pairs != expected_pairs:
        raise RuntimeError(f"group seed coverage is incomplete: {json_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    csv_pairs = {
        (int(row["weight_seed"]), int(row["point_seed"]))
        for row in csv_rows
    }
    if (
        len(csv_rows) != expected_case_count
        or csv_pairs != expected_pairs
        or any(
            row["status"] != "completed"
            or row["architecture"] != spec.architecture
            or int(row["input_dim"]) != spec.input_dim
            or int(row["state_dim"]) != spec.state_dim
            or int(row["degree"]) != degree
            for row in csv_rows
        )
    ):
        raise RuntimeError(f"group CSV count is invalid: {csv_path}")

    matrices = torch.load(
        matrix_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(matrices, dict) or set(matrices) != expected_run_ids:
        raise RuntimeError(f"group matrix keys are invalid: {matrix_path}")
    for run_id, matrix in matrices.items():
        if (
            not isinstance(matrix, torch.Tensor)
            or tuple(matrix.shape) != expected_shapes[run_id]
            or matrix.dtype != torch.float64
            or not torch.isfinite(matrix).all()
        ):
            raise RuntimeError(
                f"group matrix is invalid for run {run_id}: {matrix_path}"
            )


def _run_stage_b() -> tuple[Path, ...]:
    STAGE_B_DIR.mkdir(parents=True, exist_ok=True)
    group_json_paths: list[Path] = []
    group_count = sum(len(spec.degrees) for spec in REPLICATION_SPECS)
    completed_group_count = 0

    for spec in REPLICATION_SPECS:
        for degree in spec.degrees:
            stem = _group_stem(spec, degree)
            json_path = STAGE_B_DIR / f"{stem}.json"
            csv_path = STAGE_B_DIR / f"{stem}.csv"
            matrix_path = STAGE_B_DIR / f"{stem}_matrices.pt"
            manifest_path = STAGE_B_DIR / f"{stem}_integrity.json"
            group_json_paths.append(json_path)
            artifact_exists = (
                json_path.is_file(),
                csv_path.is_file(),
                matrix_path.is_file(),
                manifest_path.is_file(),
            )
            if any(artifact_exists) and not all(artifact_exists):
                raise RuntimeError(
                    f"incomplete resume artifacts for group {stem}"
                )
            if all(artifact_exists):
                _validate_completed_group(
                    spec,
                    degree,
                    json_path,
                    csv_path,
                    matrix_path,
                    manifest_path,
                )
                completed_group_count += 1
                print(
                    f"resume group={completed_group_count}/{group_count} "
                    f"stem={stem}",
                    flush=True,
                )
                continue

            row_oversampling = _expanded_row_oversampling(
                spec.input_dim,
                spec.state_dim,
                degree,
            )
            results = []
            matrices: dict[str, torch.Tensor] = {}
            case_count = len(WEIGHT_SEEDS) * len(POINT_SEEDS)
            completed_case_count = 0
            for weight_seed in WEIGHT_SEEDS:
                for point_seed in POINT_SEEDS:
                    case = make_rank_atlas_case(
                        architecture=spec.architecture,
                        input_dim=spec.input_dim,
                        state_dim=spec.state_dim,
                        degree=degree,
                        weight_seed=weight_seed,
                        point_seed=point_seed,
                        row_oversampling=row_oversampling,
                        relative_tolerances=RELATIVE_TOLERANCES,
                    )
                    result = _run_replication_case(case)
                    if result.evaluation_matrix is None:
                        raise RuntimeError(
                            f"replication case did not complete: {result.skip_reason}"
                        )
                    metadata = {
                        "architecture": case.architecture,
                        "input_dim": case.input_dim,
                        "state_dim": case.state_dim,
                        "degree": case.degree,
                        "weight_seed": case.weight_seed,
                        "point_seed": case.point_seed,
                    }
                    matrices[_run_id("stage_b", metadata)] = (
                        result.evaluation_matrix.detach().cpu().clone()
                    )
                    results.append(result)
                    completed_case_count += 1
                    print(
                        f"group={completed_group_count + 1}/{group_count} "
                        f"case={completed_case_count}/{case_count} "
                        f"stem={stem} weight_seed={weight_seed} "
                        f"point_seed={point_seed}",
                        flush=True,
                    )

            write_rank_atlas(
                tuple(results),
                STAGE_B_DIR,
                stem=stem,
            )
            torch.save(matrices, matrix_path)
            _write_group_integrity_manifest(
                stem,
                json_path,
                csv_path,
                matrix_path,
                manifest_path,
            )
            completed_group_count += 1
            del results, matrices
            gc.collect()

    return tuple(group_json_paths)


def _rank_tuple(result: dict[str, Any]) -> tuple[int, ...]:
    diagnostics = result["diagnostics"]
    if diagnostics is None:
        return ()
    return (
        diagnostics["numerical_rank"],
        *(
            rank["numerical_rank"]
            for rank in diagnostics["relative_ranks"]
        ),
    )


def _classification(result: dict[str, Any]) -> str:
    if result["status"] == "skipped":
        return "skipped_budget"
    diagnostics = result["diagnostics"]
    if diagnostics is None:
        return "other_failure"
    ranks = _rank_tuple(result)
    if len(set(ranks)) > 1:
        return "tolerance_sensitive"
    if ranks[0] == result["metadata"]["column_count"]:
        return "clear_full"
    return "clear_deficient"


def _health_metrics(result: dict[str, Any]) -> dict[str, Any]:
    architecture = result["metadata"]["architecture"]
    parameters = result["sampled_parameters"]
    point_values = result["evaluation_points"]
    if parameters is None or point_values is None:
        return {}

    points = torch.tensor(point_values, dtype=torch.float64)
    if architecture in (
        "shallow_biased_tanh",
        "shallow_bias_free_tanh",
    ):
        weights = torch.tensor(parameters["weights"], dtype=torch.float64)
        preactivations = torch.einsum(
            "fij,pj->fpi",
            weights,
            points,
        )
        if architecture == "shallow_biased_tanh":
            biases = torch.tensor(
                parameters["biases"],
                dtype=torch.float64,
            )
            preactivations = preactivations + biases[:, None, :]
        derivative_factors = 1.0 - torch.tanh(preactivations).square()
        count = derivative_factors.numel()
        machine_epsilon = torch.finfo(torch.float64).eps
        return {
            "tanh_max_abs_preactivation": float(
                torch.max(torch.abs(preactivations)).item()
            ),
            "tanh_min_derivative_factor": float(
                torch.min(derivative_factors).item()
            ),
            "tanh_fraction_derivative_le_1e_8": float(
                torch.count_nonzero(derivative_factors <= 1e-8).item()
                / count
            ),
            "tanh_fraction_derivative_le_1e_10": float(
                torch.count_nonzero(derivative_factors <= 1e-10).item()
                / count
            ),
            "tanh_fraction_derivative_le_1e_12": float(
                torch.count_nonzero(derivative_factors <= 1e-12).item()
                / count
            ),
            "tanh_fraction_derivative_le_machine_epsilon": float(
                torch.count_nonzero(
                    derivative_factors <= machine_epsilon
                ).item()
                / count
            ),
            "tanh_fraction_derivative_exactly_zero": float(
                torch.count_nonzero(derivative_factors == 0.0).item()
                / count
            ),
        }

    outer_weights = torch.tensor(
        parameters["outer_weights"],
        dtype=torch.float64,
    )
    diagonal_entries = torch.tensor(
        parameters["diagonal_entries"],
        dtype=torch.float64,
    )
    inner_arguments = diagonal_entries[:, None, :] * points[None, :, :]
    inner_values = torch.exp(inner_arguments)
    outer_arguments = torch.einsum(
        "fij,fpj->fpi",
        outer_weights,
        inner_values,
    )
    float64_log_max = math.log(torch.finfo(torch.float64).max)
    maximum_argument = max(
        float(torch.max(inner_arguments).item()),
        float(torch.max(outer_arguments).item()),
    )
    return {
        "exp_inner_argument_min": float(
            torch.min(inner_arguments).item()
        ),
        "exp_inner_argument_max": float(
            torch.max(inner_arguments).item()
        ),
        "exp_outer_argument_min": float(
            torch.min(outer_arguments).item()
        ),
        "exp_outer_argument_max": float(
            torch.max(outer_arguments).item()
        ),
        "exp_float64_log_max": float64_log_max,
        "exp_overflow_margin": float64_log_max - maximum_argument,
    }


def _augment_runs(
    stage_a_payload: dict[str, Any],
    stage_b_payloads: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    stage_a_lookup = {
        (
            result["metadata"]["architecture"],
            result["metadata"]["input_dim"],
            result["metadata"]["state_dim"],
            result["metadata"]["degree"],
        ): result
        for result in stage_a_payload["results"]
    }
    augmented: list[dict[str, Any]] = []

    def append_result(result: dict[str, Any], stage: str) -> None:
        metadata = result["metadata"]
        key = (
            metadata["architecture"],
            metadata["input_dim"],
            metadata["state_dim"],
            metadata["degree"],
        )
        baseline = stage_a_lookup[key]
        diagnostics = result["diagnostics"]
        sigma_ratio = None
        if (
            diagnostics is not None
            and diagnostics["largest_singular_value"] > 0.0
        ):
            sigma_ratio = (
                diagnostics["smallest_singular_value"]
                / diagnostics["largest_singular_value"]
            )
        exact_baseline_match = (
            stage == "stage_b"
            and metadata["weight_seed"]
            == baseline["metadata"]["weight_seed"]
            and metadata["point_seed"]
            == baseline["metadata"]["point_seed"]
        )
        strict_point_prefix_verified = None
        if exact_baseline_match:
            baseline_points = baseline["evaluation_points"]
            expanded_prefix = result["evaluation_points"][
                : len(baseline_points)
            ]
            strict_point_prefix_verified = (
                expanded_prefix == baseline_points
            )
            if not strict_point_prefix_verified:
                raise RuntimeError(
                    "expanded points do not preserve the Stage A prefix."
                )
        result["study"] = {
            "run_id": _run_id(stage, metadata),
            "stage": stage,
            "classification": _classification(result),
            "sigma_min_over_sigma_max": sigma_ratio,
            "baseline_point_count": baseline["metadata"]["point_count"],
            "expanded_point_count": (
                metadata["point_count"] if stage == "stage_b" else None
            ),
            "extension_point_seed": (
                _extension_point_seed(metadata["point_seed"])
                if stage == "stage_b"
                else None
            ),
            "strict_point_prefix_verified": (
                strict_point_prefix_verified
            ),
            "more_points_changed_rank": (
                _rank_tuple(result) != _rank_tuple(baseline)
                if exact_baseline_match
                else None
            ),
            "numerical_health": _health_metrics(result),
        }
        augmented.append(result)

    for result in stage_a_payload["results"]:
        append_result(result, "stage_a")
    for payload in stage_b_payloads:
        for result in payload["results"]:
            append_result(result, "stage_b")
    return augmented


def _point_seed_aggregates(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in runs:
        if result["study"]["stage"] != "stage_b":
            continue
        metadata = result["metadata"]
        key = (
            metadata["architecture"],
            metadata["input_dim"],
            metadata["state_dim"],
            metadata["degree"],
            metadata["weight_seed"],
        )
        groups[key].append(result)

    aggregates = []
    for key, group in sorted(groups.items()):
        rank_tuples = [_rank_tuple(result) for result in group]
        ratios = [
            result["study"]["sigma_min_over_sigma_max"]
            for result in group
        ]
        conditions = [
            result["diagnostics"]["retained_condition_number"]
            for result in group
        ]
        finite_ratios = [
            value for value in ratios if value is not None
        ]
        finite_conditions = [
            value for value in conditions if value is not None
        ]
        aggregates.append(
            {
                "architecture": key[0],
                "input_dim": key[1],
                "state_dim": key[2],
                "degree": key[3],
                "weight_seed": key[4],
                "point_seed_count": len(group),
                "rank_tuples": [list(ranks) for ranks in rank_tuples],
                "point_seed_rank_consensus": (
                    len(set(rank_tuples)) == 1
                ),
                "sigma_ratio_min": (
                    min(finite_ratios) if finite_ratios else None
                ),
                "sigma_ratio_median": (
                    statistics.median(finite_ratios)
                    if finite_ratios
                    else None
                ),
                "sigma_ratio_max": (
                    max(finite_ratios) if finite_ratios else None
                ),
                "condition_number_min": (
                    min(finite_conditions)
                    if finite_conditions
                    else None
                ),
                "condition_number_median": (
                    statistics.median(finite_conditions)
                    if finite_conditions
                    else None
                ),
                "condition_number_max": (
                    max(finite_conditions)
                    if finite_conditions
                    else None
                ),
            }
        )
    return aggregates


def _weight_seed_aggregates(
    point_aggregates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for aggregate in point_aggregates:
        key = (
            aggregate["architecture"],
            aggregate["input_dim"],
            aggregate["state_dim"],
            aggregate["degree"],
        )
        groups[key].append(aggregate)

    output = []
    for key, group in sorted(groups.items()):
        consensus_ranks = [
            tuple(item["rank_tuples"][0])
            if item["point_seed_rank_consensus"]
            else None
            for item in group
        ]
        witt = witt_dimension(key[1], key[3])
        ratio_minima = [
            item["sigma_ratio_min"]
            for item in group
            if item["sigma_ratio_min"] is not None
        ]
        ratio_maxima = [
            item["sigma_ratio_max"]
            for item in group
            if item["sigma_ratio_max"] is not None
        ]
        condition_minima = [
            item["condition_number_min"]
            for item in group
            if item["condition_number_min"] is not None
        ]
        condition_maxima = [
            item["condition_number_max"]
            for item in group
            if item["condition_number_max"] is not None
        ]
        output.append(
            {
                "architecture": key[0],
                "input_dim": key[1],
                "state_dim": key[2],
                "degree": key[3],
                "weight_seed_count": len(group),
                "point_seed_consensus_count": sum(
                    ranks is not None for ranks in consensus_ranks
                ),
                "weight_seed_rank_consensus": (
                    None not in consensus_ranks
                    and len(set(consensus_ranks)) == 1
                ),
                "weight_seeds_full_at_all_tolerances": sum(
                    ranks is not None
                    and all(rank == witt for rank in ranks)
                    for ranks in consensus_ranks
                ),
                "consensus_rank_tuples": [
                    list(ranks) if ranks is not None else None
                    for ranks in consensus_ranks
                ],
                "worst_sigma_ratio": (
                    min(ratio_minima) if ratio_minima else None
                ),
                "best_sigma_ratio": (
                    max(ratio_maxima) if ratio_maxima else None
                ),
                "condition_number_min": (
                    min(condition_minima) if condition_minima else None
                ),
                "condition_number_max": (
                    max(condition_maxima) if condition_maxima else None
                ),
            }
        )
    return output


def _load_stage_b_matrices() -> dict[str, torch.Tensor]:
    matrices: dict[str, torch.Tensor] = {}
    for matrix_path in sorted(STAGE_B_DIR.glob("*_matrices.pt")):
        payload = torch.load(
            matrix_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Stage B matrix payload is not a dictionary: {matrix_path}"
            )
        duplicate_run_ids = set(matrices).intersection(payload)
        if duplicate_run_ids:
            raise RuntimeError(
                "duplicate Stage B matrix run IDs: "
                f"{sorted(duplicate_run_ids)}"
            )
        matrices.update(payload)
    return matrices


def _common_nullspace_analysis(
    runs: list[dict[str, Any]],
    matrices: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in runs:
        if result["study"]["stage"] != "stage_b":
            continue
        metadata = result["metadata"]
        key = (
            metadata["architecture"],
            metadata["input_dim"],
            metadata["state_dim"],
            metadata["degree"],
        )
        groups[key].append(result)

    analyses = []
    for key, group in sorted(groups.items()):
        if all(
            all(
                rank == result["metadata"]["column_count"]
                for rank in _rank_tuple(result)
            )
            for result in group
        ):
            continue
        group_matrices = [
            matrices[result["study"]["run_id"]]
            for result in group
        ]
        normalized = []
        for matrix in group_matrices:
            spectral_norm = torch.linalg.matrix_norm(matrix, ord=2)
            normalized.append(
                matrix
                if float(spectral_norm.item()) == 0.0
                else matrix / spectral_norm
            )
        stacked = torch.cat(normalized, dim=0)
        _, singular_values, right_vectors_transposed = torch.linalg.svd(
            stacked,
            full_matrices=False,
        )
        largest = float(singular_values[0].item())
        standard_relative_tolerance = (
            max(stacked.shape) * torch.finfo(torch.float64).eps
        )
        requested_tolerances = (
            ("standard", standard_relative_tolerance),
            *(
                (f"relative_{relative_tolerance:.0e}", relative_tolerance)
                for relative_tolerance in RELATIVE_TOLERANCES
            ),
        )
        tolerance_results = []
        for tolerance_name, relative_tolerance in requested_tolerances:
            absolute_tolerance = relative_tolerance * largest
            rank = int(
                torch.count_nonzero(
                    singular_values > absolute_tolerance
                ).item()
            )
            nullspace = right_vectors_transposed[rank:, :].mT
            maximum_relative_residual = None
            if nullspace.shape[1] > 0:
                residuals = [
                    float(
                        torch.linalg.matrix_norm(
                            matrix @ nullspace,
                            ord=2,
                        ).item()
                    )
                    for matrix in normalized
                ]
                maximum_relative_residual = max(residuals)
            tolerance_results.append(
                {
                    "tolerance_name": tolerance_name,
                    "relative_tolerance": relative_tolerance,
                    "absolute_tolerance": absolute_tolerance,
                    "rank": rank,
                    "nullity": stacked.shape[1] - rank,
                    "common_nullspace_coefficients": (
                        nullspace.tolist()
                        if nullspace.shape[1] > 0
                        else []
                    ),
                    "maximum_relative_residual": (
                        maximum_relative_residual
                    ),
                }
            )
        analyses.append(
            {
                "architecture": key[0],
                "input_dim": key[1],
                "state_dim": key[2],
                "degree": key[3],
                "run_count": len(group),
                "normalization": (
                    "each nonzero matrix divided by its spectral norm; "
                    "zero matrices left unchanged"
                ),
                "stacked_shape": list(stacked.shape),
                "stacked_singular_values": singular_values.tolist(),
                "tolerance_results": tolerance_results,
                "classification": "reproducible numerical evidence only",
            }
        )
    return analyses


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _export_selected_target(
    runs: list[dict[str, Any]],
    matrices: dict[str, torch.Tensor],
) -> dict[str, Any]:
    biased_groups: dict[tuple[int, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for result in runs:
        if (
            result["study"]["stage"] == "stage_b"
            and result["metadata"]["architecture"]
            == "shallow_biased_tanh"
        ):
            metadata = result["metadata"]
            biased_groups[
                (
                    metadata["input_dim"],
                    metadata["state_dim"],
                    metadata["degree"],
                )
            ].append(result)
    reproducibly_deficient = [
        key
        for key, group in biased_groups.items()
        if len(group) == len(WEIGHT_SEEDS) * len(POINT_SEEDS)
        and all(
            result["study"]["classification"] == "clear_deficient"
            for result in group
        )
    ]
    if reproducibly_deficient:
        raise RuntimeError(
            "a replicated shallow-biased deficiency requires branch-B "
            f"analysis before target export: {reproducibly_deficient}"
        )

    biased_runs = [
        result
        for result in runs
        if result["study"]["stage"] == "stage_b"
        and result["metadata"]["architecture"] == "shallow_biased_tanh"
        and result["metadata"]["input_dim"] == 2
        and result["metadata"]["state_dim"] == 1
    ]
    by_degree: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in biased_runs:
        by_degree[result["metadata"]["degree"]].append(result)
    if set(by_degree) != {1, 2, 3, 4}:
        raise RuntimeError("selected through-degree target was not replicated.")
    if not all(
        result["study"]["classification"] == "clear_full"
        for degree_runs in by_degree.values()
        for result in degree_runs
    ):
        raise RuntimeError(
            "the predeclared branch-A target was not reproducibly full."
        )

    target_result = max(
        by_degree[4],
        key=lambda result: result["study"]["sigma_min_over_sigma_max"],
    )
    run_id = target_result["study"]["run_id"]
    matrix = matrices[run_id].numpy()
    column_count = matrix.shape[1]
    _, _, pivots = scipy.linalg.qr(
        matrix.T,
        mode="economic",
        pivoting=True,
    )
    selected_rows = np.asarray(pivots[:column_count], dtype=np.int64)
    minor = matrix[selected_rows, :]
    minor_singular_values = np.linalg.svd(
        minor,
        compute_uv=False,
    )
    determinant_sign, log_absolute_determinant = np.linalg.slogdet(minor)
    state_dim = target_result["metadata"]["state_dim"]
    point_indices = selected_rows // state_dim
    state_coordinates = selected_rows % state_dim

    SELECTED_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    parameters = target_result["sampled_parameters"]
    parameter_path = SELECTED_TARGET_DIR / "parameters.npz"
    np.savez(
        parameter_path,
        weights=np.asarray(parameters["weights"], dtype=np.float64),
        biases=np.asarray(parameters["biases"], dtype=np.float64),
    )
    points_path = SELECTED_TARGET_DIR / "evaluation_points.npy"
    np.save(
        points_path,
        np.asarray(
            target_result["evaluation_points"],
            dtype=np.float64,
        ),
    )
    matrix_path = SELECTED_TARGET_DIR / "evaluation_matrix.npy"
    np.save(matrix_path, matrix)
    spectrum_path = SELECTED_TARGET_DIR / "singular_values.npy"
    np.save(
        spectrum_path,
        np.asarray(
            target_result["diagnostics"]["singular_values"],
            dtype=np.float64,
        ),
    )
    basis_path = SELECTED_TARGET_DIR / "basis_labels.json"
    _write_json(basis_path, target_result["basis_labels"])
    minor_path = SELECTED_TARGET_DIR / "candidate_minor.npz"
    np.savez(
        minor_path,
        matrix=minor,
        selected_scalar_rows=selected_rows,
        point_indices=point_indices,
        state_coordinates=state_coordinates,
        singular_values=minor_singular_values,
    )
    minor_json_path = SELECTED_TARGET_DIR / "candidate_minor.json"
    minor_payload = {
        "classification": "reproducible numerical evidence",
        "selection_method": "pivoted QR of evaluation_matrix.T",
        "selected_scalar_rows": selected_rows.tolist(),
        "point_indices": point_indices.tolist(),
        "state_coordinates": state_coordinates.tolist(),
        "basis_labels": target_result["basis_labels"],
        "shape": list(minor.shape),
        "singular_values": minor_singular_values.tolist(),
        "condition_number": float(
            minor_singular_values[0] / minor_singular_values[-1]
        ),
        "determinant_sign": float(determinant_sign),
        "log_absolute_determinant": float(log_absolute_determinant),
        "warning": (
            "This floating-point minor is a numerical certificate target, "
            "not an exact certificate."
        ),
    }
    _write_json(minor_json_path, minor_payload)

    artifact_paths = (
        parameter_path,
        points_path,
        matrix_path,
        spectrum_path,
        basis_path,
        minor_path,
        minor_json_path,
    )
    relative_artifacts = {
        path.name: {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(path),
        }
        for path in artifact_paths
    }
    target = {
        "branch": "A",
        "classification": "reproducible numerical evidence",
        "architecture": "shallow_biased_tanh",
        "input_dim": 2,
        "state_dim": 1,
        "through_degree": 4,
        "target_degree": 4,
        "weight_seed": target_result["metadata"]["weight_seed"],
        "point_seed": target_result["metadata"]["point_seed"],
        "point_count": target_result["metadata"]["point_count"],
        "extension_point_seed": target_result["study"][
            "extension_point_seed"
        ],
        "row_oversampling": target_result["metadata"]["row_oversampling"],
        "dtype": "float64",
        "matrix_shape": list(matrix.shape),
        "witt_dimension": column_count,
        "parameter_scales": target_result["metadata"]["parameter_scales"],
        "standard_tolerance": target_result["diagnostics"][
            "standard_tolerance"
        ],
        "relative_tolerances": [
            rank["relative_tolerance"]
            for rank in target_result["diagnostics"]["relative_ranks"]
        ],
        "bracket_convention": (
            "[V,W](x) = DW(x)V(x) - DV(x)W(x)"
        ),
        "basis": "complete standard Lyndon basis of degree 4",
        "rank_tuple": list(_rank_tuple(target_result)),
        "sigma_min_over_sigma_max": (
            target_result["study"]["sigma_min_over_sigma_max"]
        ),
        "artifacts": relative_artifacts,
        "minor": minor_payload,
        "warning": (
            "Branch A selects an exact-certificate target; it does not claim "
            "generic faithfulness."
        ),
    }
    manifest_path = SELECTED_TARGET_DIR / "manifest.json"
    _write_json(manifest_path, target)
    target["manifest"] = {
        "path": manifest_path.relative_to(REPO_ROOT).as_posix(),
        "sha256": _sha256(manifest_path),
    }
    return target


CSV_FIELDS = (
    "run_id",
    "stage",
    "classification",
    "architecture",
    "input_dim",
    "state_dim",
    "degree",
    "weight_seed",
    "point_seed",
    "point_count",
    "row_count",
    "column_count",
    "basis_count",
    "row_oversampling",
    "actual_row_oversampling",
    "matrix_entry_count",
    "dtype",
    "point_distribution",
    "point_scale",
    "weight_scale",
    "bias_scale",
    "outer_weight_scale",
    "diagonal_scale",
    "status",
    "failure_kind",
    "failure_message",
    "elapsed_seconds",
    "numerical_rank_standard",
    "nullity_standard",
    "standard_tolerance",
    "relative_tolerance_1e_8",
    "absolute_tolerance_1e_8",
    "rank_rtol_1e_8",
    "relative_tolerance_1e_10",
    "absolute_tolerance_1e_10",
    "rank_rtol_1e_10",
    "relative_tolerance_1e_12",
    "absolute_tolerance_1e_12",
    "rank_rtol_1e_12",
    "rank_min",
    "rank_max",
    "tolerance_sensitive",
    "sigma_max",
    "sigma_min",
    "sigma_min_over_sigma_max",
    "smallest_retained_singular_value",
    "retained_condition_number",
    "smallest_column_norm",
    "largest_column_norm",
    "baseline_point_count",
    "expanded_point_count",
    "extension_point_seed",
    "strict_point_prefix_verified",
    "more_points_changed_rank",
    "tanh_max_abs_preactivation",
    "tanh_min_derivative_factor",
    "tanh_fraction_derivative_le_1e_8",
    "tanh_fraction_derivative_le_1e_10",
    "tanh_fraction_derivative_le_1e_12",
    "tanh_fraction_derivative_le_machine_epsilon",
    "tanh_fraction_derivative_exactly_zero",
    "exp_inner_argument_min",
    "exp_inner_argument_max",
    "exp_outer_argument_min",
    "exp_outer_argument_max",
    "exp_float64_log_max",
    "exp_overflow_margin",
)


def _csv_row(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result["metadata"]
    diagnostics = result["diagnostics"]
    study = result["study"]
    health = study["numerical_health"]
    parameter_scales = metadata["parameter_scales"]
    relative_ranks = (
        diagnostics["relative_ranks"] if diagnostics is not None else []
    )
    ranks = _rank_tuple(result)
    row = {
        "run_id": study["run_id"],
        "stage": study["stage"],
        "classification": study["classification"],
        "architecture": metadata["architecture"],
        "input_dim": metadata["input_dim"],
        "state_dim": metadata["state_dim"],
        "degree": metadata["degree"],
        "weight_seed": metadata["weight_seed"],
        "point_seed": metadata["point_seed"],
        "point_count": metadata["point_count"],
        "row_count": metadata["row_count"],
        "column_count": metadata["column_count"],
        "basis_count": len(result["basis_labels"]),
        "row_oversampling": metadata["row_oversampling"],
        "actual_row_oversampling": metadata["actual_row_oversampling"],
        "matrix_entry_count": metadata["matrix_entry_count"],
        "dtype": metadata["dtype"],
        "point_distribution": metadata["point_distribution"],
        "point_scale": metadata["point_scale"],
        "weight_scale": parameter_scales.get("weight_scale", ""),
        "bias_scale": parameter_scales.get("bias_scale", ""),
        "outer_weight_scale": parameter_scales.get(
            "outer_weight_scale",
            "",
        ),
        "diagonal_scale": parameter_scales.get("diagonal_scale", ""),
        "status": result["status"],
        "failure_kind": "",
        "failure_message": result["skip_reason"] or "",
        "elapsed_seconds": result["elapsed_seconds"],
        "numerical_rank_standard": (
            diagnostics["numerical_rank"] if diagnostics else ""
        ),
        "nullity_standard": (
            diagnostics["nullity"] if diagnostics else ""
        ),
        "standard_tolerance": (
            diagnostics["standard_tolerance"] if diagnostics else ""
        ),
        "rank_min": min(ranks) if ranks else "",
        "rank_max": max(ranks) if ranks else "",
        "tolerance_sensitive": (
            len(set(ranks)) > 1 if ranks else ""
        ),
        "sigma_max": (
            diagnostics["largest_singular_value"] if diagnostics else ""
        ),
        "sigma_min": (
            diagnostics["smallest_singular_value"] if diagnostics else ""
        ),
        "sigma_min_over_sigma_max": (
            study["sigma_min_over_sigma_max"]
            if study["sigma_min_over_sigma_max"] is not None
            else ""
        ),
        "retained_condition_number": (
            diagnostics["retained_condition_number"] if diagnostics else ""
        ),
        "smallest_retained_singular_value": (
            diagnostics["smallest_retained_singular_value"]
            if diagnostics
            else ""
        ),
        "smallest_column_norm": (
            diagnostics["smallest_column_norm"] if diagnostics else ""
        ),
        "largest_column_norm": (
            diagnostics["largest_column_norm"] if diagnostics else ""
        ),
        "baseline_point_count": study["baseline_point_count"],
        "expanded_point_count": (
            study["expanded_point_count"]
            if study["expanded_point_count"] is not None
            else ""
        ),
        "extension_point_seed": (
            study["extension_point_seed"]
            if study["extension_point_seed"] is not None
            else ""
        ),
        "strict_point_prefix_verified": (
            study["strict_point_prefix_verified"]
            if study["strict_point_prefix_verified"] is not None
            else ""
        ),
        "more_points_changed_rank": (
            study["more_points_changed_rank"]
            if study["more_points_changed_rank"] is not None
            else ""
        ),
    }
    relative_fields = (
        (
            "relative_tolerance_1e_8",
            "absolute_tolerance_1e_8",
            "rank_rtol_1e_8",
        ),
        (
            "relative_tolerance_1e_10",
            "absolute_tolerance_1e_10",
            "rank_rtol_1e_10",
        ),
        (
            "relative_tolerance_1e_12",
            "absolute_tolerance_1e_12",
            "rank_rtol_1e_12",
        ),
    )
    for index, fields in enumerate(relative_fields):
        if index < len(relative_ranks):
            relative_rank = relative_ranks[index]
            row[fields[0]] = relative_rank["relative_tolerance"]
            row[fields[1]] = relative_rank["absolute_tolerance"]
            row[fields[2]] = relative_rank["numerical_rank"]
        else:
            for field in fields:
                row[field] = ""
    for field in CSV_FIELDS:
        if field not in row:
            row[field] = health.get(field, "")
    return row


def _write_combined_csv(runs: list[dict[str, Any]]) -> Path:
    csv_path = OUTPUT_DIR / "rank_atlas.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(_csv_row(result) for result in runs)
    return csv_path


def main() -> int:
    """Run Stage B and write combined, reproducible research artifacts."""
    if not STAGE_A_JSON.is_file() or not STAGE_A_CSV.is_file():
        raise FileNotFoundError(
            "Stage A artifacts are missing; run the full atlas first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    group_json_paths = _run_stage_b()
    stage_a_payload = _load_json(STAGE_A_JSON)
    stage_b_payloads = tuple(
        _load_json(path) for path in group_json_paths
    )
    runs = _augment_runs(stage_a_payload, stage_b_payloads)
    point_aggregates = _point_seed_aggregates(runs)
    weight_aggregates = _weight_seed_aggregates(point_aggregates)
    matrices = _load_stage_b_matrices()
    nullspace_analyses = _common_nullspace_analysis(runs, matrices)
    selected_target = _export_selected_target(runs, matrices)
    csv_path = _write_combined_csv(runs)

    classification_counts: dict[str, int] = defaultdict(int)
    for result in runs:
        classification_counts[result["study"]["classification"]] += 1
    combined_payload = {
        "schema_version": 1,
        "evidence_kind": "numerical_homogeneous_rank_only",
        "architecture_formulas": {
            "shallow_biased_tanh": "V_i(x) = tanh(A_i x + b_i)",
            "shallow_bias_free_tanh": "V_i(x) = tanh(A_i x)",
            "depth_two_exponential": (
                "V_i(x) = exp(A_i exp(D_i x)); D_i is diagonal"
            ),
        },
        "bracket_convention": (
            "[V,W](x) = DW(x)V(x) - DV(x)W(x)"
        ),
        "basis": "standard Lyndon basis, one homogeneous degree at a time",
        "witt_dimensions": {
            "input_dim_2": {"1": 2, "2": 1, "3": 2, "4": 3},
            "input_dim_3": {"1": 3, "2": 3, "3": 8, "4": 18},
        },
        "runtime_metadata": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": str(np.__version__),
            "scipy_version": str(scipy.__version__),
            "torch_version": str(torch.__version__),
            "torch_num_threads": torch.get_num_threads(),
            "torch_config": torch.__config__.show(),
        },
        "stage_a_protocol": {
            "weight_seeds": [1729],
            "point_seeds": [2718],
            "row_oversampling": DEFAULT_ROW_OVERSAMPLING,
            "source_json": STAGE_A_JSON.relative_to(REPO_ROOT).as_posix(),
            "run_count": len(stage_a_payload["results"]),
        },
        "stage_b_protocol": {
            "weight_seeds": list(WEIGHT_SEEDS),
            "point_seeds": list(POINT_SEEDS),
            "point_count_rule": "twice the matching Stage A point count",
            "point_construction": (
                "exact Stage A point prefix followed by independent normal "
                "points from local seed 1_000_000_000 + point_seed"
            ),
            "relative_tolerances": list(RELATIVE_TOLERANCES),
            "specifications": [
                {
                    "architecture": spec.architecture,
                    "input_dim": spec.input_dim,
                    "state_dim": spec.state_dim,
                    "degrees": list(spec.degrees),
                }
                for spec in REPLICATION_SPECS
            ],
            "run_count": sum(
                len(payload["results"]) for payload in stage_b_payloads
            ),
        },
        "classification_rule": {
            "clear_full": (
                "standard and every configured relative rank equal "
                "the Witt dimension"
            ),
            "clear_deficient": (
                "all ranks agree below the Witt dimension"
            ),
            "tolerance_sensitive": (
                "standard and configured ranks do not all agree"
            ),
        },
        "classification_counts": dict(classification_counts),
        "run_count": len(runs),
        "runs": runs,
        "point_seed_aggregates": point_aggregates,
        "weight_seed_aggregates": weight_aggregates,
        "candidate_common_nullspaces": nullspace_analyses,
        "selected_target": selected_target,
        "conclusion_ledger": [
            {
                "classification": "exact fact from the algebra module",
                "statement": (
                    "Witt dimensions are (2,1,2,3) for input_dim=2 "
                    "and (3,3,8,18) for input_dim=3 through degree 4."
                ),
            },
            {
                "classification": "reproducible numerical evidence",
                "statement": (
                    "The selected branch-A target is full on the complete "
                    "homogeneous Lyndon bases in every replicated run."
                ),
            },
            {
                "classification": "unproved conjecture",
                "statement": (
                    "A later exact nonzero-minor calculation may certify "
                    "generic finite-degree faithfulness at the selected target."
                ),
            },
        ],
    }
    json_path = OUTPUT_DIR / "rank_atlas.json"
    _write_json(json_path, combined_payload)
    print(
        f"combined_runs={len(runs)} csv={csv_path} json={json_path}",
        flush=True,
    )
    print(
        f"selected_branch={selected_target['branch']} "
        f"selected_run={selected_target['manifest']['path']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
