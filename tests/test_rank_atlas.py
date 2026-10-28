"""Tests for the numerical homogeneous free-Lie rank atlas.

Runs with pytest (``pytest tests``) or as a script
(``python tests/test_rank_atlas.py``).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Callable
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lie_reservoir.rank_atlas as rank_atlas_module  # noqa: E402
from lie_reservoir import (  # noqa: E402
    DepthTwoExponentialParameters,
    ShallowBiasedTanhConfig,
    ShallowBiasedTanhParameters,
    ShallowBiasFreeTanhParameters,
    build_homogeneous_evaluation_matrix,
    compute_rank_diagnostics,
    format_rank_table,
    full_atlas_cases,
    make_rank_atlas_case,
    make_vector_fields,
    run_rank_atlas,
    run_rank_case,
    sample_evaluation_points,
    sample_parameters,
    smoke_atlas_cases,
    witt_dimension,
    write_rank_atlas,
)
from lie_reservoir.vector_fields import Parameters  # noqa: E402


def _assert_raises(
    exception_type: type[Exception],
    function: Callable[..., object],
    *args: object,
) -> None:
    try:
        function(*args)
    except exception_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"expected {exception_type.__name__}, got {type(exc).__name__}"
        ) from exc
    raise AssertionError(f"expected {exception_type.__name__}")


def _parameter_tensors(parameters: Parameters) -> tuple[torch.Tensor, ...]:
    if isinstance(parameters, ShallowBiasedTanhParameters):
        return parameters.weights, parameters.biases
    if isinstance(parameters, ShallowBiasFreeTanhParameters):
        return (parameters.weights,)
    if not isinstance(parameters, DepthTwoExponentialParameters):
        raise TypeError("unsupported parameter container.")
    return parameters.outer_weights, parameters.diagonal_entries


def _parameters_equal(
    left: Parameters,
    right: Parameters,
) -> bool:
    return type(left) is type(right) and all(
        torch.equal(left_tensor, right_tensor)
        for left_tensor, right_tensor in zip(
            _parameter_tensors(left),
            _parameter_tensors(right),
            strict=True,
        )
    )


def _load_research_script(module_name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "lie_rank_research.py"
    )
    specification = importlib.util.spec_from_file_location(
        module_name,
        script_path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_matrix_shape_witt_columns_and_degree_one_layout() -> None:
    cases = (
        make_rank_atlas_case(
            "shallow_biased_tanh",
            input_dim=2,
            state_dim=2,
            degree=4,
            row_oversampling=1,
        ),
        make_rank_atlas_case(
            "shallow_biased_tanh",
            input_dim=3,
            state_dim=3,
            degree=3,
            row_oversampling=2,
        ),
    )
    for case in cases:
        parameters = sample_parameters(
            case.vector_field_config,
            seed=case.weight_seed,
        )
        points = sample_evaluation_points(case)
        matrix = build_homogeneous_evaluation_matrix(
            case.vector_field_config,
            parameters,
            case.degree,
            points,
        )
        assert matrix.shape == (
            case.point_count * case.state_dim,
            witt_dimension(case.input_dim, case.degree),
        )
        assert matrix.shape[1] == case.column_count
        assert matrix.dtype == torch.float64
        assert torch.isfinite(matrix).all()
        diagnostics = compute_rank_diagnostics(matrix)
        assert diagnostics.singular_values.shape == (case.column_count,)
        assert diagnostics.column_norms.shape == (case.column_count,)
        assert torch.isfinite(diagnostics.singular_values).all()
        assert torch.all(
            diagnostics.singular_values[:-1]
            >= diagnostics.singular_values[1:]
        )
        assert diagnostics.numerical_rank + diagnostics.nullity == (
            case.column_count
        )
        assert diagnostics.right_numerical_nullspace.shape == (
            case.column_count,
            diagnostics.nullity,
        )

    assert witt_dimension(2, 3) == 2
    degree_one = make_rank_atlas_case(
        "shallow_biased_tanh",
        input_dim=3,
        state_dim=2,
        degree=1,
        row_oversampling=2,
    )
    parameters = sample_parameters(
        degree_one.vector_field_config,
        seed=degree_one.weight_seed,
    )
    points = sample_evaluation_points(degree_one)
    actual = build_homogeneous_evaluation_matrix(
        degree_one.vector_field_config,
        parameters,
        degree_one.degree,
        points,
    )
    vector_fields = make_vector_fields(
        degree_one.vector_field_config,
        parameters,
    )
    expected = torch.cat(
        tuple(
            torch.stack(
                tuple(vector_field(point) for vector_field in vector_fields),
                dim=1,
            )
            for point in points.unbind(dim=0)
        ),
        dim=0,
    )
    assert torch.equal(actual, expected)


def test_seed_reproducibility_and_isolation() -> None:
    base_case = make_rank_atlas_case(
        "shallow_biased_tanh",
        input_dim=2,
        state_dim=2,
        degree=2,
        row_oversampling=4,
        weight_seed=101,
        point_seed=202,
    )
    global_random_state = torch.random.get_rng_state().clone()
    default_dtype = torch.get_default_dtype()
    first = run_rank_case(base_case)
    repeated = run_rank_case(base_case)
    changed_points = run_rank_case(
        make_rank_atlas_case(
            "shallow_biased_tanh",
            input_dim=2,
            state_dim=2,
            degree=2,
            row_oversampling=4,
            weight_seed=101,
            point_seed=203,
        )
    )
    changed_weights = run_rank_case(
        make_rank_atlas_case(
            "shallow_biased_tanh",
            input_dim=2,
            state_dim=2,
            degree=2,
            row_oversampling=4,
            weight_seed=102,
            point_seed=202,
        )
    )

    assert first.parameters is not None
    assert repeated.parameters is not None
    assert changed_points.parameters is not None
    assert changed_weights.parameters is not None
    assert first.evaluation_points is not None
    assert repeated.evaluation_points is not None
    assert changed_points.evaluation_points is not None
    assert changed_weights.evaluation_points is not None
    assert first.evaluation_matrix is not None
    assert repeated.evaluation_matrix is not None
    assert changed_points.evaluation_matrix is not None
    assert changed_weights.evaluation_matrix is not None
    assert first.diagnostics is not None
    assert repeated.diagnostics is not None

    assert _parameters_equal(first.parameters, repeated.parameters)
    assert torch.equal(first.evaluation_points, repeated.evaluation_points)
    assert torch.equal(first.evaluation_matrix, repeated.evaluation_matrix)
    assert torch.equal(
        first.diagnostics.singular_values,
        repeated.diagnostics.singular_values,
    )
    assert first.diagnostics.numerical_rank == (
        repeated.diagnostics.numerical_rank
    )

    assert _parameters_equal(first.parameters, changed_points.parameters)
    assert not torch.equal(
        first.evaluation_points,
        changed_points.evaluation_points,
    )
    assert not torch.equal(
        first.evaluation_matrix,
        changed_points.evaluation_matrix,
    )

    assert not _parameters_equal(first.parameters, changed_weights.parameters)
    assert torch.equal(
        first.evaluation_points,
        changed_weights.evaluation_points,
    )
    assert not torch.equal(
        first.evaluation_matrix,
        changed_weights.evaluation_matrix,
    )
    assert torch.equal(global_random_state, torch.random.get_rng_state())
    assert torch.get_default_dtype() == default_dtype


def test_one_parameter_object_is_reused_across_all_points() -> None:
    case = make_rank_atlas_case(
        "shallow_biased_tanh",
        input_dim=2,
        state_dim=2,
        degree=2,
        row_oversampling=6,
    )
    parameters = sample_parameters(
        case.vector_field_config,
        seed=case.weight_seed,
    )
    points = sample_evaluation_points(case)
    assert points.shape[0] > 1

    with patch.object(
        rank_atlas_module,
        "make_vector_fields",
        wraps=rank_atlas_module.make_vector_fields,
    ) as make_fields_spy:
        matrix = build_homogeneous_evaluation_matrix(
            case.vector_field_config,
            parameters,
            case.degree,
            points,
        )
    assert matrix.shape[0] == case.point_count * case.state_dim
    assert make_fields_spy.call_count == 1
    assert make_fields_spy.call_args.args[1] is parameters

    with (
        patch.object(
            rank_atlas_module,
            "sample_parameters",
            wraps=rank_atlas_module.sample_parameters,
        ) as sample_spy,
        patch.object(
            rank_atlas_module,
            "lyndon_basis",
            wraps=rank_atlas_module.lyndon_basis,
        ) as basis_spy,
    ):
        result = run_rank_case(case)
    assert result.status == "completed"
    assert sample_spy.call_count == 1
    assert basis_spy.call_count == 1


def test_rank_diagnostics_known_full_and_deficient_examples() -> None:
    duplicated = torch.tensor(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        dtype=torch.float64,
    )
    deficient = compute_rank_diagnostics(
        duplicated,
        relative_tolerances=(1e-8, 1e-12),
    )
    assert deficient.numerical_rank == 1
    assert deficient.nullity == 1
    assert deficient.numerical_rank <= min(duplicated.shape)
    assert deficient.right_numerical_nullspace.shape == (2, 1)
    expected_column_norm = math.sqrt(14.0)
    torch.testing.assert_close(
        deficient.column_norms,
        torch.full((2,), expected_column_norm, dtype=torch.float64),
    )
    assert math.isclose(
        deficient.smallest_column_norm,
        expected_column_norm,
        rel_tol=1e-15,
    )
    assert math.isclose(
        deficient.largest_column_norm,
        expected_column_norm,
        rel_tol=1e-15,
    )
    torch.testing.assert_close(
        duplicated @ deficient.right_numerical_nullspace,
        torch.zeros((3, 1), dtype=torch.float64),
        rtol=0.0,
        atol=1e-14,
    )
    pivot_rows = torch.argmax(
        torch.abs(deficient.right_numerical_nullspace),
        dim=0,
    )
    pivot_values = deficient.right_numerical_nullspace[
        pivot_rows,
        torch.arange(deficient.nullity),
    ]
    assert torch.all(pivot_values >= 0)

    independent = torch.eye(3, dtype=torch.float64)
    full = compute_rank_diagnostics(
        independent,
        relative_tolerances=(1e-8, 1e-12),
    )
    assert full.numerical_rank == 3
    assert full.nullity == 0
    assert full.right_numerical_nullspace.shape == (3, 0)
    expected_tolerance = (
        max(independent.shape)
        * torch.finfo(torch.float64).eps
        * full.largest_singular_value
    )
    assert full.standard_tolerance == expected_tolerance

    tolerance_probe = torch.diag(
        torch.tensor([1.0, 1e-9], dtype=torch.float64)
    )
    probe = compute_rank_diagnostics(
        tolerance_probe,
        relative_tolerances=(1e-8, 1e-10),
    )
    assert tuple(rank.numerical_rank for rank in probe.relative_ranks) == (
        1,
        2,
    )

    boundary_probe = compute_rank_diagnostics(
        torch.diag(torch.tensor([1.0, 0.5], dtype=torch.float64)),
        relative_tolerances=(0.5,),
    )
    assert boundary_probe.relative_ranks[0].absolute_tolerance == 0.5
    assert boundary_probe.relative_ranks[0].numerical_rank == 1

    machine_epsilon = torch.finfo(torch.float64).eps
    standard_boundary = compute_rank_diagnostics(
        torch.diag(
            torch.tensor(
                [1.0, 2.0 * machine_epsilon],
                dtype=torch.float64,
            )
        )
    )
    assert standard_boundary.standard_tolerance == 2.0 * machine_epsilon
    assert standard_boundary.numerical_rank == 1


def test_manual_degree_one_architecture_example_is_full_rank() -> None:
    config = ShallowBiasedTanhConfig(input_dim=2, state_dim=2)
    parameters = ShallowBiasedTanhParameters(
        weights=torch.zeros((2, 2, 2), dtype=torch.float64),
        biases=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float64,
        ),
    )
    points = torch.zeros((1, 2), dtype=torch.float64)
    matrix = build_homogeneous_evaluation_matrix(
        config,
        parameters,
        degree=1,
        evaluation_points=points,
    )
    expected = (
        torch.eye(2, dtype=torch.float64)
        * torch.tanh(torch.tensor(1.0, dtype=torch.float64))
    )
    assert torch.equal(matrix, expected)
    diagnostics = compute_rank_diagnostics(matrix)
    assert diagnostics.numerical_rank == 2
    assert diagnostics.nullity == 0


def test_end_to_end_constant_fields_are_exactly_rank_deficient() -> None:
    case = make_rank_atlas_case(
        "shallow_biased_tanh",
        input_dim=2,
        state_dim=2,
        degree=2,
        row_oversampling=2,
        shallow_weight_scale=0.0,
    )
    result = run_rank_case(case)
    assert result.status == "completed"
    assert result.evaluation_matrix is not None
    assert result.diagnostics is not None
    diagnostics = result.diagnostics

    assert torch.count_nonzero(result.evaluation_matrix) == 0
    assert torch.count_nonzero(diagnostics.singular_values) == 0
    assert torch.count_nonzero(diagnostics.column_norms) == 0
    assert diagnostics.largest_singular_value == 0.0
    assert diagnostics.smallest_singular_value == 0.0
    assert diagnostics.standard_tolerance == 0.0
    assert diagnostics.numerical_rank == 0
    assert diagnostics.nullity == case.column_count
    assert result.numerically_full_column_rank is False
    assert diagnostics.smallest_retained_singular_value is None
    assert diagnostics.retained_condition_number is None
    assert all(rank.numerical_rank == 0 for rank in diagnostics.relative_ranks)
    assert all(
        rank.absolute_tolerance == 0.0
        for rank in diagnostics.relative_ranks
    )
    assert diagnostics.right_numerical_nullspace.shape == (
        case.column_count,
        case.column_count,
    )
    assert torch.count_nonzero(
        result.evaluation_matrix
        @ diagnostics.right_numerical_nullspace
    ) == 0

    with tempfile.TemporaryDirectory() as temporary_directory:
        _csv_path, json_path = write_rank_atlas(
            (result,),
            Path(temporary_directory),
            stem="zero_rank",
        )
        with json_path.open("r", encoding="utf-8") as json_file:
            payload = json.load(json_file)
    persisted = payload["results"][0]
    assert persisted["diagnostics"]["numerical_rank"] == 0
    assert persisted["diagnostics"]["nullity"] == case.column_count
    assert persisted["diagnostics"]["retained_condition_number"] is None


def test_budget_skip_occurs_before_sampling_or_autodiff() -> None:
    case = make_rank_atlas_case(
        "shallow_biased_tanh",
        input_dim=2,
        state_dim=2,
        degree=1,
        row_oversampling=0,
    )
    assert case.row_count * case.column_count == 4
    with (
        patch.object(
            rank_atlas_module,
            "sample_parameters",
            side_effect=AssertionError("sampling must not run"),
        ),
        patch.object(
            rank_atlas_module,
            "_build_evaluation_matrix_from_basis",
            side_effect=AssertionError("autodiff must not run"),
        ),
        patch.object(
            rank_atlas_module,
            "lyndon_basis",
            side_effect=AssertionError("basis generation must not run"),
        ),
    ):
        skipped = run_rank_case(case, max_matrix_entries=3)
    assert skipped.status == "skipped"
    assert skipped.diagnostics is None
    assert skipped.parameters is None
    assert skipped.evaluation_points is None
    assert skipped.evaluation_matrix is None
    assert skipped.basis_labels == ()
    assert skipped.numerically_full_column_rank is None
    assert skipped.skip_reason is not None
    assert "4" in skipped.skip_reason and "3" in skipped.skip_reason

    completed = run_rank_case(case, max_matrix_entries=4)
    assert completed.status == "completed"
    table = format_rank_table((skipped, completed))
    table_lines = table.splitlines()
    assert len(table_lines) == 4
    assert "cond_image" in table_lines[0]
    assert table_lines[2].split()[-1] == "skipped"
    assert "-" in table_lines[2].split()
    assert table_lines[3].split()[-1] == "completed"
    assert "2/2" in table_lines[3].split()


def test_json_csv_counts_and_tolerances_agree() -> None:
    case = make_rank_atlas_case(
        "shallow_bias_free_tanh",
        input_dim=2,
        state_dim=2,
        degree=1,
        row_oversampling=0,
        relative_tolerances=(1.25e-11, 3.5e-9),
    )
    completed = run_rank_case(case, max_matrix_entries=4)
    skipped = run_rank_case(case, max_matrix_entries=3)
    results = (completed, skipped)

    with tempfile.TemporaryDirectory() as temporary_directory:
        csv_path, json_path = write_rank_atlas(
            results,
            Path(temporary_directory),
            stem="atlas_test",
        )
        with json_path.open("r", encoding="utf-8") as json_file:
            payload = json.load(json_file)
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            csv_rows = list(csv.DictReader(csv_file))

    assert payload["result_count"] == len(results)
    assert len(payload["results"]) == len(csv_rows) == len(results)
    assert payload["completed_count"] == 1
    assert payload["skipped_count"] == 1
    assert payload["device"] == "cpu"
    assert payload["platform"]
    assert payload["torch_config"]
    assert "may rotate" in payload["nullspace_basis_note"]
    completed_payload = payload["results"][0]
    skipped_payload = payload["results"][1]
    completed_metadata = completed_payload["metadata"]
    completed_diagnostics = completed_payload["diagnostics"]
    skipped_metadata = skipped_payload["metadata"]
    assert completed_payload["metadata"]["relative_tolerances"] == [
        1.25e-11,
        3.5e-9,
    ]
    assert completed_payload["metadata"]["standard_tolerance"] is not None
    assert completed_payload["diagnostics"]["singular_values"]
    assert (
        completed_payload["diagnostics"]["right_numerical_nullspace"]
        is not None
    )
    assert completed_payload["sampled_parameters"] is not None
    assert completed_payload["evaluation_points"] is not None
    assert skipped_payload["diagnostics"] is None
    assert skipped_payload["numerically_full_column_rank"] is None
    completed_csv = csv_rows[0]
    skipped_csv = csv_rows[1]
    assert completed_csv["architecture"] == completed_metadata["architecture"]
    for field in (
        "input_dim",
        "state_dim",
        "degree",
        "weight_seed",
        "point_seed",
        "point_count",
        "row_count",
        "column_count",
        "row_oversampling",
        "actual_row_oversampling",
        "matrix_entry_count",
        "max_matrix_entries",
    ):
        assert int(completed_csv[field]) == completed_metadata[field]
    for field in (
        "standard_relative_tolerance",
        "standard_tolerance",
    ):
        assert float(completed_csv[field]) == completed_metadata[field]
    for field in (
        "numerical_rank",
        "nullity",
    ):
        assert int(completed_csv[field]) == completed_diagnostics[field]
    assert completed_csv["numerically_full_column_rank"] == str(
        completed_payload["numerically_full_column_rank"]
    ).lower()
    for field in (
        "largest_singular_value",
        "smallest_singular_value",
        "largest_column_norm",
        "smallest_column_norm",
        "smallest_retained_singular_value",
        "retained_condition_number",
    ):
        assert float(completed_csv[field]) == completed_diagnostics[field]
    assert len(completed_diagnostics["column_norms"]) == (
        completed_metadata["column_count"]
    )
    assert completed_csv["device"] == "cpu"
    for index, relative_rank in enumerate(
        completed_diagnostics["relative_ranks"],
        start=1,
    ):
        assert float(completed_csv[f"relative_tolerance_{index}"]) == (
            relative_rank["relative_tolerance"]
        )
        assert float(completed_csv[f"absolute_tolerance_{index}"]) == (
            relative_rank["absolute_tolerance"]
        )
        assert int(completed_csv[f"numerical_rank_{index}"]) == (
            relative_rank["numerical_rank"]
        )
        assert int(completed_csv[f"nullity_{index}"]) == (
            relative_rank["nullity"]
        )
    assert csv_rows[1]["status"] == "skipped"
    assert skipped_csv["skip_reason"] == skipped_payload["skip_reason"]
    assert int(skipped_csv["matrix_entry_count"]) == (
        skipped_metadata["matrix_entry_count"]
    )
    assert int(skipped_csv["max_matrix_entries"]) == (
        skipped_metadata["max_matrix_entries"]
    )
    for field in (
        "standard_relative_tolerance",
        "standard_tolerance",
        "numerical_rank",
        "nullity",
        "numerically_full_column_rank",
        "largest_singular_value",
        "smallest_singular_value",
        "largest_column_norm",
        "smallest_column_norm",
        "smallest_retained_singular_value",
        "retained_condition_number",
        "absolute_tolerance_1",
        "numerical_rank_1",
        "nullity_1",
        "absolute_tolerance_2",
        "numerical_rank_2",
        "nullity_2",
    ):
        assert skipped_csv[field] == ""


def test_default_grids_are_homogeneous_and_tiny_tests_do_not_run_full() -> None:
    smoke_cases = smoke_atlas_cases()
    full_cases = full_atlas_cases()
    assert len(smoke_cases) == 6
    assert len(full_cases) == 120
    assert {
        (case.architecture, case.input_dim, case.state_dim, case.degree)
        for case in smoke_cases
    } == {
        (architecture, 2, 2, degree)
        for architecture in (
            "shallow_biased_tanh",
            "shallow_bias_free_tanh",
            "depth_two_exponential",
        )
        for degree in (1, 2)
    }
    full_keys = {
        (case.architecture, case.input_dim, case.state_dim, case.degree)
        for case in full_cases
    }
    expected_full_keys = {
        (architecture, input_dim, state_dim, degree)
        for architecture in (
            "shallow_biased_tanh",
            "shallow_bias_free_tanh",
            "depth_two_exponential",
        )
        for input_dim in (2, 3)
        for degree in (1, 2, 3, 4)
        for state_dim in (1, 2, 3, 4, 6)
    }
    assert full_keys == expected_full_keys
    assert len(full_keys) == len(full_cases)
    assert all(
        case.row_count >= case.column_count + case.row_oversampling
        for case in full_cases
    )
    assert all(
        case.row_count >= 2 * case.column_count
        for case in full_cases
    )

    smoke_results = run_rank_atlas(smoke_cases)
    assert len(smoke_results) == 6
    assert all(result.status == "completed" for result in smoke_results)
    assert all(
        result.diagnostics is not None
        and result.diagnostics.numerical_rank
        <= min(result.case.row_count, result.case.column_count)
        for result in smoke_results
    )


def test_single_configuration_cli_writes_one_record() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "lie_rank_atlas.py"
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        completed_process = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "single",
                "--architecture",
                "shallow_biased_tanh",
                "--input-dim",
                "2",
                "--state-dim",
                "2",
                "--degree",
                "1",
                "--weight-seed",
                "3",
                "--point-seed",
                "5",
                "--row-oversampling",
                "0",
                "--relative-tolerances",
                "1e-8,1e-10",
                "--output-dir",
                temporary_directory,
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed_process.returncode == 0, completed_process.stderr
        json_path = Path(temporary_directory) / "single_rank_atlas.json"
        csv_path = Path(temporary_directory) / "single_rank_atlas.csv"
        assert json_path.is_file()
        assert csv_path.is_file()
        with json_path.open("r", encoding="utf-8") as json_file:
            payload = json.load(json_file)
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            csv_rows = list(csv.DictReader(csv_file))

    assert payload["result_count"] == 1
    assert len(csv_rows) == 1
    result_payload = payload["results"][0]
    metadata = result_payload["metadata"]
    assert result_payload["status"] == "completed"
    assert metadata["architecture"] == "shallow_biased_tanh"
    assert metadata["input_dim"] == 2
    assert metadata["state_dim"] == 2
    assert metadata["degree"] == 1
    assert metadata["weight_seed"] == 3
    assert metadata["point_seed"] == 5
    assert metadata["relative_tolerances"] == [
        1e-8,
        1e-10,
    ]
    assert result_payload["diagnostics"]["numerical_rank"] <= (
        metadata["column_count"]
    )
    assert csv_rows[0]["status"] == "completed"
    assert int(csv_rows[0]["row_count"]) == metadata["row_count"]
    assert int(csv_rows[0]["column_count"]) == metadata["column_count"]
    assert "architecture" in completed_process.stdout
    assert "numerical homogeneous-rank evidence only" in (
        completed_process.stdout
    )


def test_cli_repo_root_search_fails_loudly_when_package_is_absent() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "lie_rank_atlas.py"
    )
    specification = importlib.util.spec_from_file_location(
        "lie_rank_atlas_cli_root_test",
        script_path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as temporary_directory:
        module.__file__ = str(
            Path(temporary_directory) / "scripts" / "lie_rank_atlas.py"
        )
        _assert_raises(RuntimeError, module._find_repo_root)


def test_research_replication_appends_the_exact_baseline_points() -> None:
    module = _load_research_script(
        "lie_rank_research_point_test",
    )

    row_oversampling = module._expanded_row_oversampling(2, 1, 4)
    expanded_case = make_rank_atlas_case(
        "shallow_biased_tanh",
        input_dim=2,
        state_dim=1,
        degree=4,
        weight_seed=1729,
        point_seed=2718,
        row_oversampling=row_oversampling,
    )
    baseline_case = make_rank_atlas_case(
        "shallow_biased_tanh",
        input_dim=2,
        state_dim=1,
        degree=4,
        weight_seed=1729,
        point_seed=2718,
    )
    baseline_points = sample_evaluation_points(baseline_case)
    first = module._sample_expanded_points(expanded_case)
    repeated = module._sample_expanded_points(expanded_case)

    assert first.shape[0] == 2 * baseline_points.shape[0]
    assert torch.equal(
        first[: baseline_points.shape[0]],
        baseline_points,
    )
    assert torch.equal(first, repeated)


def test_research_resume_rejects_stale_and_duplicate_artifacts() -> None:
    module = _load_research_script(
        "lie_rank_research_resume_test",
    )
    replication_spec = module.ReplicationSpec(
        architecture="shallow_biased_tanh",
        input_dim=2,
        state_dim=1,
        degrees=(1,),
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory)
        json_path = output_dir / "group.json"
        csv_path = output_dir / "group.csv"
        matrix_path = output_dir / "group_matrices.pt"
        manifest_path = output_dir / "group_integrity.json"
        json_path.write_text(
            json.dumps(
                {
                    "results": [],
                    "completed_count": 0,
                    "skipped_count": 0,
                }
            ),
            encoding="utf-8",
        )
        csv_path.write_text("status\n", encoding="utf-8")
        torch.save({}, matrix_path)
        module._write_group_integrity_manifest(
            "shallow_biased_tanh_input2_state1_degree1",
            json_path,
            csv_path,
            matrix_path,
            manifest_path,
        )

        _assert_raises(
            RuntimeError,
            module._validate_completed_group,
            replication_spec,
            1,
            json_path,
            csv_path,
            matrix_path,
            manifest_path,
        )

        matrix_path.write_bytes(matrix_path.read_bytes() + b"stale")
        _assert_raises(
            RuntimeError,
            module._validate_completed_group,
            replication_spec,
            1,
            json_path,
            csv_path,
            matrix_path,
            manifest_path,
        )

        duplicate_matrix = torch.ones((1, 1), dtype=torch.float64)
        torch.save(
            {"duplicate_run": duplicate_matrix},
            output_dir / "first_matrices.pt",
        )
        torch.save(
            {"duplicate_run": duplicate_matrix},
            output_dir / "second_matrices.pt",
        )
        with patch.object(module, "STAGE_B_DIR", output_dir):
            _assert_raises(
                RuntimeError,
                module._load_stage_b_matrices,
            )


def test_research_zero_matrix_evidence_remains_reportable() -> None:
    module = _load_research_script(
        "lie_rank_research_zero_test",
    )
    runs = []
    matrices = {}
    for point_seed in module.POINT_SEEDS:
        run_id = f"zero_{point_seed}"
        result = {
            "metadata": {
                "architecture": "shallow_biased_tanh",
                "input_dim": 2,
                "state_dim": 1,
                "degree": 2,
                "column_count": 1,
                "weight_seed": module.WEIGHT_SEEDS[0],
                "point_seed": point_seed,
            },
            "diagnostics": {
                "numerical_rank": 0,
                "retained_condition_number": None,
                "relative_ranks": [
                    {
                        "relative_tolerance": relative_tolerance,
                        "numerical_rank": 0,
                    }
                    for relative_tolerance in module.RELATIVE_TOLERANCES
                ],
            },
            "study": {
                "stage": "stage_b",
                "run_id": run_id,
                "sigma_min_over_sigma_max": None,
            },
        }
        runs.append(result)
        matrices[run_id] = torch.zeros((4, 1), dtype=torch.float64)

    point_aggregates = module._point_seed_aggregates(runs)
    assert len(point_aggregates) == 1
    assert point_aggregates[0]["sigma_ratio_min"] is None
    assert point_aggregates[0]["condition_number_min"] is None
    weight_aggregates = module._weight_seed_aggregates(point_aggregates)
    assert weight_aggregates[0]["worst_sigma_ratio"] is None
    assert weight_aggregates[0]["condition_number_min"] is None

    nullspace_analyses = module._common_nullspace_analysis(runs, matrices)
    assert len(nullspace_analyses) == 1
    for tolerance_result in nullspace_analyses[0]["tolerance_results"]:
        assert tolerance_result["rank"] == 0
        assert tolerance_result["nullity"] == 1
        assert tolerance_result["maximum_relative_residual"] == 0.0


def test_validation_rejects_malformed_inputs() -> None:
    _assert_raises(
        ValueError,
        make_rank_atlas_case,
        "unsupported",
        2,
        2,
        1,
    )
    _assert_raises(
        ValueError,
        lambda: make_rank_atlas_case(
            "shallow_biased_tanh",
            2,
            2,
            1,
            relative_tolerances=(1e-8, 1e-8),
        ),
    )
    case = make_rank_atlas_case(
        "shallow_biased_tanh",
        2,
        2,
        1,
    )
    parameters = sample_parameters(
        case.vector_field_config,
        seed=case.weight_seed,
    )
    _assert_raises(
        ValueError,
        build_homogeneous_evaluation_matrix,
        case.vector_field_config,
        parameters,
        case.degree,
        torch.zeros((1, 3), dtype=torch.float64),
    )
    _assert_raises(
        TypeError,
        build_homogeneous_evaluation_matrix,
        case.vector_field_config,
        parameters,
        case.degree,
        torch.zeros((1, 2), dtype=torch.float32),
    )
    _assert_raises(
        ValueError,
        compute_rank_diagnostics,
        torch.tensor([[float("nan")]], dtype=torch.float64),
    )
    _assert_raises(
        ValueError,
        compute_rank_diagnostics,
        torch.ones((1, 2), dtype=torch.float64),
    )
    _assert_raises(
        ValueError,
        lambda: run_rank_case(case, max_matrix_entries=0),
    )


def _run_all() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
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
