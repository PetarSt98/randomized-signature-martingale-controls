"""Numerical rank diagnostics for homogeneous represented free-Lie spaces.

Every :class:`RankAtlasCase` contains one exact homogeneous ``degree``. The
evaluation matrix never concatenates degrees. Its rows are ordered first by
evaluation point and then by state coordinate; its columns follow the standard
Lyndon basis order from :func:`lyndon_basis`.

The resulting ranks and nullspaces are floating-point evidence only. They are
not exact certificates or exact Lie identities.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Literal, TypeAlias

import torch

from .brackets import make_lie_evaluator
from .free_lie import LieNode, format_lie_node, lyndon_basis, witt_dimension
from .vector_fields import (
    DepthTwoExponentialConfig,
    DepthTwoExponentialParameters,
    Parameters,
    ShallowBiasedTanhConfig,
    ShallowBiasedTanhParameters,
    ShallowBiasFreeTanhConfig,
    ShallowBiasFreeTanhParameters,
    VectorFieldConfig,
    make_vector_fields,
    sample_parameters,
)


ArchitectureName: TypeAlias = Literal[
    "shallow_biased_tanh",
    "shallow_bias_free_tanh",
    "depth_two_exponential",
]

ARCHITECTURES: tuple[ArchitectureName, ...] = (
    "shallow_biased_tanh",
    "shallow_bias_free_tanh",
    "depth_two_exponential",
)
DEFAULT_INPUT_DIMS = (2, 3)
DEFAULT_DEGREES = (1, 2, 3, 4)
DEFAULT_STATE_DIMS = (1, 2, 3, 4, 6)
DEFAULT_RELATIVE_TOLERANCES = (1e-8, 1e-10, 1e-12)
DEFAULT_ROW_OVERSAMPLING = witt_dimension(
    max(DEFAULT_INPUT_DIMS),
    max(DEFAULT_DEGREES),
)
DEFAULT_POINT_SCALE = 1.0
DEFAULT_WEIGHT_SEED = 1729
DEFAULT_POINT_SEED = 2718
DEFAULT_MAX_MATRIX_ENTRIES = 1_000_000

_SUPPORTED_CONFIG_TYPES = (
    ShallowBiasedTanhConfig,
    ShallowBiasFreeTanhConfig,
    DepthTwoExponentialConfig,
)


def _validate_integer(
    name: str,
    value: int,
    *,
    positive: bool,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    lower_bound = 1 if positive else 0
    if value < lower_bound:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}.")


def _validate_seed(name: str, seed: int) -> None:
    _validate_integer(name, seed, positive=False)
    if seed > 2**63 - 1:
        raise ValueError(f"{name} must be at most 2**63 - 1.")


def _validate_positive_scale(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _validate_relative_tolerances(
    relative_tolerances: tuple[float, ...],
) -> None:
    if not isinstance(relative_tolerances, tuple):
        raise TypeError("relative_tolerances must be a tuple.")
    if not relative_tolerances:
        raise ValueError("relative_tolerances must not be empty.")
    converted: list[float] = []
    for relative_tolerance in relative_tolerances:
        if (
            isinstance(relative_tolerance, bool)
            or not isinstance(relative_tolerance, Real)
        ):
            raise TypeError("every relative tolerance must be a real number.")
        converted_tolerance = float(relative_tolerance)
        if (
            not math.isfinite(converted_tolerance)
            or converted_tolerance <= 0
            or converted_tolerance >= 1
        ):
            raise ValueError(
                "every relative tolerance must be finite and lie in (0, 1)."
            )
        converted.append(converted_tolerance)
    if len(set(converted)) != len(converted):
        raise ValueError("relative_tolerances must be unique.")


def architecture_name(config: VectorFieldConfig) -> ArchitectureName:
    """Return the canonical persisted architecture name."""
    if isinstance(config, ShallowBiasedTanhConfig):
        return "shallow_biased_tanh"
    if isinstance(config, ShallowBiasFreeTanhConfig):
        return "shallow_bias_free_tanh"
    if isinstance(config, DepthTwoExponentialConfig):
        return "depth_two_exponential"
    raise TypeError("config must be a supported vector-field configuration.")


def parameter_scales(config: VectorFieldConfig) -> dict[str, float]:
    """Return only the numerical scales used by ``config``."""
    if isinstance(config, ShallowBiasedTanhConfig):
        return {
            "weight_scale": float(config.weight_scale),
            "bias_scale": float(config.bias_scale),
        }
    if isinstance(config, ShallowBiasFreeTanhConfig):
        return {"weight_scale": float(config.weight_scale)}
    if isinstance(config, DepthTwoExponentialConfig):
        return {
            "outer_weight_scale": float(config.outer_weight_scale),
            "diagonal_scale": float(config.diagonal_scale),
        }
    raise TypeError("config must be a supported vector-field configuration.")


@dataclass(frozen=True)
class RankAtlasCase:
    """One architecture draw and one homogeneous Lie degree.

    ``weight_seed`` follows the research metadata terminology and seeds the
    complete architecture parameter draw, including biases or diagonal entries.
    """

    vector_field_config: VectorFieldConfig
    degree: int
    weight_seed: int
    point_seed: int
    row_oversampling: int = DEFAULT_ROW_OVERSAMPLING
    point_scale: float = DEFAULT_POINT_SCALE
    relative_tolerances: tuple[float, ...] = DEFAULT_RELATIVE_TOLERANCES

    def __post_init__(self) -> None:
        if not isinstance(self.vector_field_config, _SUPPORTED_CONFIG_TYPES):
            raise TypeError(
                "vector_field_config must be a supported configuration."
            )
        _validate_integer("degree", self.degree, positive=True)
        _validate_seed("weight_seed", self.weight_seed)
        _validate_seed("point_seed", self.point_seed)
        _validate_integer(
            "row_oversampling",
            self.row_oversampling,
            positive=False,
        )
        _validate_positive_scale("point_scale", self.point_scale)
        _validate_relative_tolerances(self.relative_tolerances)
        if self.column_count == 0:
            raise ValueError(
                "the requested homogeneous free-Lie space has dimension zero."
            )

    @property
    def architecture(self) -> ArchitectureName:
        return architecture_name(self.vector_field_config)

    @property
    def input_dim(self) -> int:
        return self.vector_field_config.input_dim

    @property
    def state_dim(self) -> int:
        return self.vector_field_config.state_dim

    @property
    def column_count(self) -> int:
        return witt_dimension(self.input_dim, self.degree)

    @property
    def point_count(self) -> int:
        required_rows = self.column_count + self.row_oversampling
        return max(1, math.ceil(required_rows / self.state_dim))

    @property
    def row_count(self) -> int:
        return self.point_count * self.state_dim

    @property
    def actual_row_oversampling(self) -> int:
        return self.row_count - self.column_count


@dataclass(frozen=True)
class RelativeRank:
    """Rank at one user-requested relative tolerance."""

    relative_tolerance: float
    absolute_tolerance: float
    numerical_rank: int
    nullity: int


@dataclass(frozen=True, eq=False)
class RankDiagnostics:
    """Complete float64 SVD diagnostics for one homogeneous matrix."""

    singular_values: torch.Tensor
    column_norms: torch.Tensor
    largest_singular_value: float
    smallest_singular_value: float
    largest_column_norm: float
    smallest_column_norm: float
    machine_epsilon: float
    standard_relative_tolerance: float
    standard_tolerance: float
    numerical_rank: int
    nullity: int
    smallest_retained_singular_value: float | None
    retained_condition_number: float | None
    right_numerical_nullspace: torch.Tensor
    relative_ranks: tuple[RelativeRank, ...]


@dataclass(frozen=True, eq=False)
class RankAtlasResult:
    """Completed or explicitly budget-skipped homogeneous atlas case."""

    case: RankAtlasCase
    status: Literal["completed", "skipped"]
    basis_labels: tuple[str, ...]
    matrix_entry_count: int
    max_matrix_entries: int | None
    elapsed_seconds: float
    parameters: Parameters | None
    evaluation_points: torch.Tensor | None
    evaluation_matrix: torch.Tensor | None
    diagnostics: RankDiagnostics | None
    skip_reason: str | None

    @property
    def numerically_full_column_rank(self) -> bool | None:
        if self.diagnostics is None:
            return None
        return self.diagnostics.numerical_rank == self.case.column_count


def make_vector_field_config(
    architecture: ArchitectureName,
    input_dim: int,
    state_dim: int,
    *,
    shallow_weight_scale: float = 1.0,
    shallow_bias_scale: float = 1.0,
    outer_weight_scale: float = 0.1,
    diagonal_scale: float = 0.1,
) -> VectorFieldConfig:
    """Construct one of the three supported architecture configurations."""
    if architecture == "shallow_biased_tanh":
        return ShallowBiasedTanhConfig(
            input_dim=input_dim,
            state_dim=state_dim,
            weight_scale=shallow_weight_scale,
            bias_scale=shallow_bias_scale,
        )
    if architecture == "shallow_bias_free_tanh":
        return ShallowBiasFreeTanhConfig(
            input_dim=input_dim,
            state_dim=state_dim,
            weight_scale=shallow_weight_scale,
        )
    if architecture == "depth_two_exponential":
        return DepthTwoExponentialConfig(
            input_dim=input_dim,
            state_dim=state_dim,
            outer_weight_scale=outer_weight_scale,
            diagonal_scale=diagonal_scale,
        )
    raise ValueError(f"unsupported architecture: {architecture!r}.")


def make_rank_atlas_case(
    architecture: ArchitectureName,
    input_dim: int,
    state_dim: int,
    degree: int,
    *,
    weight_seed: int = DEFAULT_WEIGHT_SEED,
    point_seed: int = DEFAULT_POINT_SEED,
    row_oversampling: int = DEFAULT_ROW_OVERSAMPLING,
    point_scale: float = DEFAULT_POINT_SCALE,
    relative_tolerances: tuple[float, ...] = DEFAULT_RELATIVE_TOLERANCES,
    shallow_weight_scale: float = 1.0,
    shallow_bias_scale: float = 1.0,
    outer_weight_scale: float = 0.1,
    diagonal_scale: float = 0.1,
) -> RankAtlasCase:
    """Build a validated homogeneous atlas case."""
    config = make_vector_field_config(
        architecture=architecture,
        input_dim=input_dim,
        state_dim=state_dim,
        shallow_weight_scale=shallow_weight_scale,
        shallow_bias_scale=shallow_bias_scale,
        outer_weight_scale=outer_weight_scale,
        diagonal_scale=diagonal_scale,
    )
    return RankAtlasCase(
        vector_field_config=config,
        degree=degree,
        weight_seed=weight_seed,
        point_seed=point_seed,
        row_oversampling=row_oversampling,
        point_scale=point_scale,
        relative_tolerances=relative_tolerances,
    )


def smoke_atlas_cases(
    **case_options: object,
) -> tuple[RankAtlasCase, ...]:
    """Return the six-case smoke grid used by tests and the smoke CLI."""
    return tuple(
        make_rank_atlas_case(
            architecture=architecture,
            input_dim=2,
            state_dim=2,
            degree=degree,
            **case_options,
        )
        for architecture in ARCHITECTURES
        for degree in (1, 2)
    )


def full_atlas_cases(
    **case_options: object,
) -> tuple[RankAtlasCase, ...]:
    """Return the unexecuted 120-case default reconnaissance grid."""
    return tuple(
        make_rank_atlas_case(
            architecture=architecture,
            input_dim=input_dim,
            state_dim=state_dim,
            degree=degree,
            **case_options,
        )
        for architecture in ARCHITECTURES
        for input_dim in DEFAULT_INPUT_DIMS
        for degree in DEFAULT_DEGREES
        for state_dim in DEFAULT_STATE_DIMS
    )


def sample_evaluation_points(case: RankAtlasCase) -> torch.Tensor:
    """Sample generic Euclidean states with a fresh local CPU generator."""
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(case.point_seed)
    return torch.randn(
        (case.point_count, case.state_dim),
        generator=random_generator,
        dtype=torch.float64,
        device="cpu",
    ) * float(case.point_scale)


def _validate_evaluation_points(
    evaluation_points: torch.Tensor,
    state_dim: int,
) -> None:
    if not isinstance(evaluation_points, torch.Tensor):
        raise TypeError("evaluation_points must be a torch.Tensor.")
    if evaluation_points.ndim != 2:
        raise ValueError("evaluation_points must be a two-dimensional tensor.")
    if evaluation_points.shape[0] == 0:
        raise ValueError("evaluation_points must contain at least one point.")
    if evaluation_points.shape[1] != state_dim:
        raise ValueError(
            f"evaluation_points must have shape (point_count, {state_dim})."
        )
    if evaluation_points.dtype != torch.float64:
        raise TypeError("evaluation_points must have dtype torch.float64.")
    if not torch.isfinite(evaluation_points).all():
        raise ValueError("evaluation_points must contain only finite values.")


def build_homogeneous_evaluation_matrix(
    config: VectorFieldConfig,
    parameters: Parameters,
    degree: int,
    evaluation_points: torch.Tensor,
) -> torch.Tensor:
    """Build one homogeneous matrix from one fixed parameter container.

    ``make_vector_fields`` is called exactly once. Every basis evaluator and
    every evaluation point therefore uses the same parameter object.
    """
    _validate_integer("degree", degree, positive=True)
    if not isinstance(config, _SUPPORTED_CONFIG_TYPES):
        raise TypeError("config must be a supported vector-field configuration.")
    _validate_evaluation_points(evaluation_points, config.state_dim)
    basis = lyndon_basis(config.input_dim, degree)
    if not basis:
        raise ValueError(
            "the requested homogeneous free-Lie space has dimension zero."
        )
    return _build_evaluation_matrix_from_basis(
        config=config,
        parameters=parameters,
        basis=basis,
        evaluation_points=evaluation_points,
    )


def _build_evaluation_matrix_from_basis(
    config: VectorFieldConfig,
    parameters: Parameters,
    basis: tuple[LieNode, ...],
    evaluation_points: torch.Tensor,
) -> torch.Tensor:
    vector_fields = make_vector_fields(config, parameters)
    evaluators = tuple(
        make_lie_evaluator(node, vector_fields)
        for node in basis
    )
    point_blocks = tuple(
        torch.stack(
            tuple(evaluator(point) for evaluator in evaluators),
            dim=1,
        )
        for point in evaluation_points.unbind(dim=0)
    )
    matrix = torch.cat(point_blocks, dim=0)
    expected_shape = (
        evaluation_points.shape[0] * config.state_dim,
        len(basis),
    )
    if tuple(matrix.shape) != expected_shape:
        raise RuntimeError(
            f"evaluation matrix has shape {tuple(matrix.shape)}, "
            f"expected {expected_shape}."
        )
    if matrix.dtype != torch.float64:
        raise RuntimeError("evaluation matrix must have dtype torch.float64.")
    if not torch.isfinite(matrix).all():
        raise FloatingPointError("evaluation matrix contains a non-finite value.")
    return matrix


def _canonicalize_nullspace_columns(
    right_nullspace: torch.Tensor,
) -> torch.Tensor:
    if right_nullspace.shape[1] == 0:
        return right_nullspace
    column_indices = torch.arange(
        right_nullspace.shape[1],
        device=right_nullspace.device,
    )
    pivot_rows = torch.argmax(torch.abs(right_nullspace), dim=0)
    pivot_values = right_nullspace[pivot_rows, column_indices]
    signs = torch.where(
        pivot_values < 0,
        -torch.ones_like(pivot_values),
        torch.ones_like(pivot_values),
    )
    return (right_nullspace * signs).contiguous()


def compute_rank_diagnostics(
    matrix: torch.Tensor,
    relative_tolerances: tuple[float, ...] = DEFAULT_RELATIVE_TOLERANCES,
) -> RankDiagnostics:
    """Compute the complete float64 singular spectrum and numerical ranks."""
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("matrix must be a torch.Tensor.")
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix must have at least one row and one column.")
    if matrix.shape[0] < matrix.shape[1]:
        raise ValueError(
            "matrix must have at least as many rows as columns; "
            "atlas cases choose enough evaluation points automatically."
        )
    if matrix.dtype != torch.float64:
        raise TypeError("matrix must have dtype torch.float64.")
    if not torch.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite values.")
    _validate_relative_tolerances(relative_tolerances)

    _, singular_values, right_vectors_transposed = torch.linalg.svd(
        matrix,
        full_matrices=False,
    )
    column_norms = torch.linalg.vector_norm(matrix, dim=0)
    largest = float(singular_values[0].item())
    smallest = float(singular_values[-1].item())
    largest_column_norm = float(torch.max(column_norms).item())
    smallest_column_norm = float(torch.min(column_norms).item())
    machine_epsilon = float(torch.finfo(torch.float64).eps)
    standard_relative_tolerance = max(matrix.shape) * machine_epsilon
    standard_tolerance = standard_relative_tolerance * largest
    numerical_rank = int(
        torch.count_nonzero(singular_values > standard_tolerance).item()
    )
    column_count = matrix.shape[1]
    nullity = column_count - numerical_rank
    smallest_retained = (
        float(singular_values[numerical_rank - 1].item())
        if numerical_rank > 0
        else None
    )
    condition_number = (
        largest / smallest_retained
        if smallest_retained is not None
        else None
    )
    right_nullspace = _canonicalize_nullspace_columns(
        right_vectors_transposed[numerical_rank:, :].mT
    )

    relative_rank_results: list[RelativeRank] = []
    for relative_tolerance in relative_tolerances:
        absolute_tolerance = float(relative_tolerance) * largest
        relative_rank = int(
            torch.count_nonzero(
                singular_values > absolute_tolerance
            ).item()
        )
        relative_rank_results.append(
            RelativeRank(
                relative_tolerance=float(relative_tolerance),
                absolute_tolerance=absolute_tolerance,
                numerical_rank=relative_rank,
                nullity=column_count - relative_rank,
            )
        )
    return RankDiagnostics(
        singular_values=singular_values,
        column_norms=column_norms,
        largest_singular_value=largest,
        smallest_singular_value=smallest,
        largest_column_norm=largest_column_norm,
        smallest_column_norm=smallest_column_norm,
        machine_epsilon=machine_epsilon,
        standard_relative_tolerance=standard_relative_tolerance,
        standard_tolerance=standard_tolerance,
        numerical_rank=numerical_rank,
        nullity=nullity,
        smallest_retained_singular_value=smallest_retained,
        retained_condition_number=condition_number,
        right_numerical_nullspace=right_nullspace,
        relative_ranks=tuple(relative_rank_results),
    )


def _validate_max_matrix_entries(
    max_matrix_entries: int | None,
) -> None:
    if max_matrix_entries is None:
        return
    _validate_integer(
        "max_matrix_entries",
        max_matrix_entries,
        positive=True,
    )


def run_rank_case(
    case: RankAtlasCase,
    *,
    max_matrix_entries: int | None = DEFAULT_MAX_MATRIX_ENTRIES,
) -> RankAtlasResult:
    """Run or explicitly budget-skip one homogeneous atlas case."""
    if not isinstance(case, RankAtlasCase):
        raise TypeError("case must be a RankAtlasCase.")
    _validate_max_matrix_entries(max_matrix_entries)
    started = time.perf_counter()
    matrix_entry_count = case.row_count * case.column_count

    if (
        max_matrix_entries is not None
        and matrix_entry_count > max_matrix_entries
    ):
        skip_reason = (
            f"matrix entry count {matrix_entry_count} exceeds "
            f"max_matrix_entries {max_matrix_entries}"
        )
        return RankAtlasResult(
            case=case,
            status="skipped",
            basis_labels=(),
            matrix_entry_count=matrix_entry_count,
            max_matrix_entries=max_matrix_entries,
            elapsed_seconds=time.perf_counter() - started,
            parameters=None,
            evaluation_points=None,
            evaluation_matrix=None,
            diagnostics=None,
            skip_reason=skip_reason,
        )

    basis = lyndon_basis(case.input_dim, case.degree)
    basis_labels = tuple(format_lie_node(node) for node in basis)
    parameters = sample_parameters(
        case.vector_field_config,
        seed=case.weight_seed,
    )
    evaluation_points = sample_evaluation_points(case)
    matrix = _build_evaluation_matrix_from_basis(
        config=case.vector_field_config,
        parameters=parameters,
        basis=basis,
        evaluation_points=evaluation_points,
    )
    diagnostics = compute_rank_diagnostics(
        matrix,
        relative_tolerances=case.relative_tolerances,
    )
    return RankAtlasResult(
        case=case,
        status="completed",
        basis_labels=basis_labels,
        matrix_entry_count=matrix_entry_count,
        max_matrix_entries=max_matrix_entries,
        elapsed_seconds=time.perf_counter() - started,
        parameters=parameters,
        evaluation_points=evaluation_points,
        evaluation_matrix=matrix,
        diagnostics=diagnostics,
        skip_reason=None,
    )


def run_rank_atlas(
    cases: tuple[RankAtlasCase, ...],
    *,
    max_matrix_entries: int | None = DEFAULT_MAX_MATRIX_ENTRIES,
) -> tuple[RankAtlasResult, ...]:
    """Run homogeneous cases sequentially without multiprocessing."""
    if not isinstance(cases, tuple):
        raise TypeError("cases must be a tuple.")
    if not cases:
        raise ValueError("cases must not be empty.")
    _validate_max_matrix_entries(max_matrix_entries)
    return tuple(
        run_rank_case(
            case,
            max_matrix_entries=max_matrix_entries,
        )
        for case in cases
    )


def _parameters_payload(parameters: Parameters) -> dict[str, object]:
    if isinstance(parameters, ShallowBiasedTanhParameters):
        return {
            "parameter_type": "shallow_biased_tanh",
            "weights": parameters.weights.tolist(),
            "biases": parameters.biases.tolist(),
        }
    if isinstance(parameters, ShallowBiasFreeTanhParameters):
        return {
            "parameter_type": "shallow_bias_free_tanh",
            "weights": parameters.weights.tolist(),
        }
    if isinstance(parameters, DepthTwoExponentialParameters):
        return {
            "parameter_type": "depth_two_exponential",
            "outer_weights": parameters.outer_weights.tolist(),
            "diagonal_entries": parameters.diagonal_entries.tolist(),
        }
    raise TypeError("parameters must be a supported parameter container.")


def _metadata_payload(result: RankAtlasResult) -> dict[str, object]:
    case = result.case
    diagnostics = result.diagnostics
    return {
        "architecture": case.architecture,
        "input_dim": case.input_dim,
        "state_dim": case.state_dim,
        "degree": case.degree,
        "weight_seed": case.weight_seed,
        "point_seed": case.point_seed,
        "point_count": case.point_count,
        "row_count": case.row_count,
        "column_count": case.column_count,
        "row_oversampling": case.row_oversampling,
        "actual_row_oversampling": case.actual_row_oversampling,
        "dtype": "torch.float64",
        "point_distribution": "normal",
        "point_scale": float(case.point_scale),
        "parameter_scales": parameter_scales(case.vector_field_config),
        "relative_tolerances": [
            float(value) for value in case.relative_tolerances
        ],
        "machine_epsilon": (
            diagnostics.machine_epsilon if diagnostics is not None else None
        ),
        "standard_relative_tolerance": (
            diagnostics.standard_relative_tolerance
            if diagnostics is not None
            else None
        ),
        "standard_tolerance": (
            diagnostics.standard_tolerance
            if diagnostics is not None
            else None
        ),
        "matrix_entry_count": result.matrix_entry_count,
        "max_matrix_entries": result.max_matrix_entries,
    }


def _diagnostics_payload(
    diagnostics: RankDiagnostics,
) -> dict[str, object]:
    return {
        "singular_values": diagnostics.singular_values.tolist(),
        "column_norms": diagnostics.column_norms.tolist(),
        "largest_singular_value": diagnostics.largest_singular_value,
        "smallest_singular_value": diagnostics.smallest_singular_value,
        "largest_column_norm": diagnostics.largest_column_norm,
        "smallest_column_norm": diagnostics.smallest_column_norm,
        "standard_tolerance": diagnostics.standard_tolerance,
        "numerical_rank": diagnostics.numerical_rank,
        "nullity": diagnostics.nullity,
        "smallest_retained_singular_value": (
            diagnostics.smallest_retained_singular_value
        ),
        "retained_condition_number": (
            diagnostics.retained_condition_number
        ),
        "right_numerical_nullspace": (
            diagnostics.right_numerical_nullspace.tolist()
        ),
        "relative_ranks": [
            {
                "relative_tolerance": rank.relative_tolerance,
                "absolute_tolerance": rank.absolute_tolerance,
                "numerical_rank": rank.numerical_rank,
                "nullity": rank.nullity,
            }
            for rank in diagnostics.relative_ranks
        ],
    }


def _result_payload(result: RankAtlasResult) -> dict[str, object]:
    completed = result.status == "completed"
    if completed:
        if (
            result.parameters is None
            or result.evaluation_points is None
            or result.diagnostics is None
        ):
            raise RuntimeError("completed result is missing persisted data.")
        parameters_payload = _parameters_payload(result.parameters)
        evaluation_points = result.evaluation_points.tolist()
        diagnostics_payload = _diagnostics_payload(result.diagnostics)
    else:
        parameters_payload = None
        evaluation_points = None
        diagnostics_payload = None

    return {
        "status": result.status,
        "skip_reason": result.skip_reason,
        "elapsed_seconds": result.elapsed_seconds,
        "metadata": _metadata_payload(result),
        "basis_labels": list(result.basis_labels),
        "matrix_shape": [result.case.row_count, result.case.column_count],
        "numerically_full_column_rank": (
            result.numerically_full_column_rank
        ),
        "sampled_parameters": parameters_payload,
        "evaluation_points": evaluation_points,
        "diagnostics": diagnostics_payload,
    }


_CSV_BASE_FIELDS = (
    "schema_version",
    "evidence_kind",
    "basis",
    "bracket_convention",
    "matrix_row_order",
    "matrix_column_order",
    "nullspace_sign_convention",
    "nullspace_basis_note",
    "status",
    "skip_reason",
    "architecture",
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
    "dtype",
    "device",
    "point_distribution",
    "point_scale",
    "weight_scale",
    "bias_scale",
    "outer_weight_scale",
    "diagonal_scale",
    "machine_epsilon",
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
    "elapsed_seconds",
)


def _csv_row(result: RankAtlasResult) -> dict[str, object]:
    case = result.case
    diagnostics = result.diagnostics
    scales = parameter_scales(case.vector_field_config)
    row: dict[str, object] = {
        "schema_version": 1,
        "evidence_kind": "numerical_rank_only",
        "basis": "standard_lyndon",
        "bracket_convention": "[V,W](x) = DW(x)V(x) - DV(x)W(x)",
        "matrix_row_order": "evaluation_point_then_state_coordinate",
        "matrix_column_order": "lexicographic_standard_lyndon_basis",
        "nullspace_sign_convention": "largest_absolute_entry_positive",
        "nullspace_basis_note": (
            "sign-canonicalized SVD basis; repeated singular subspaces "
            "may rotate across numerical backends"
        ),
        "status": result.status,
        "skip_reason": result.skip_reason or "",
        "architecture": case.architecture,
        "input_dim": case.input_dim,
        "state_dim": case.state_dim,
        "degree": case.degree,
        "weight_seed": case.weight_seed,
        "point_seed": case.point_seed,
        "point_count": case.point_count,
        "row_count": case.row_count,
        "column_count": case.column_count,
        "row_oversampling": case.row_oversampling,
        "actual_row_oversampling": case.actual_row_oversampling,
        "matrix_entry_count": result.matrix_entry_count,
        "max_matrix_entries": (
            result.max_matrix_entries
            if result.max_matrix_entries is not None
            else ""
        ),
        "dtype": "torch.float64",
        "device": "cpu",
        "point_distribution": "normal",
        "point_scale": float(case.point_scale),
        "weight_scale": scales.get("weight_scale", ""),
        "bias_scale": scales.get("bias_scale", ""),
        "outer_weight_scale": scales.get("outer_weight_scale", ""),
        "diagonal_scale": scales.get("diagonal_scale", ""),
        "machine_epsilon": (
            diagnostics.machine_epsilon if diagnostics is not None else ""
        ),
        "standard_relative_tolerance": (
            diagnostics.standard_relative_tolerance
            if diagnostics is not None
            else ""
        ),
        "standard_tolerance": (
            diagnostics.standard_tolerance
            if diagnostics is not None
            else ""
        ),
        "numerical_rank": (
            diagnostics.numerical_rank if diagnostics is not None else ""
        ),
        "nullity": diagnostics.nullity if diagnostics is not None else "",
        "numerically_full_column_rank": (
            (
                "true"
                if result.numerically_full_column_rank
                else "false"
            )
            if diagnostics is not None
            else ""
        ),
        "largest_singular_value": (
            diagnostics.largest_singular_value
            if diagnostics is not None
            else ""
        ),
        "smallest_singular_value": (
            diagnostics.smallest_singular_value
            if diagnostics is not None
            else ""
        ),
        "largest_column_norm": (
            diagnostics.largest_column_norm
            if diagnostics is not None
            else ""
        ),
        "smallest_column_norm": (
            diagnostics.smallest_column_norm
            if diagnostics is not None
            else ""
        ),
        "smallest_retained_singular_value": (
            diagnostics.smallest_retained_singular_value
            if diagnostics is not None
            else ""
        ),
        "retained_condition_number": (
            diagnostics.retained_condition_number
            if diagnostics is not None
            else ""
        ),
        "elapsed_seconds": result.elapsed_seconds,
    }
    for index, relative_tolerance in enumerate(
        case.relative_tolerances,
        start=1,
    ):
        row[f"relative_tolerance_{index}"] = relative_tolerance
        if diagnostics is None:
            row[f"absolute_tolerance_{index}"] = ""
            row[f"numerical_rank_{index}"] = ""
            row[f"nullity_{index}"] = ""
            continue
        relative_rank = diagnostics.relative_ranks[index - 1]
        row[f"absolute_tolerance_{index}"] = (
            relative_rank.absolute_tolerance
        )
        row[f"numerical_rank_{index}"] = relative_rank.numerical_rank
        row[f"nullity_{index}"] = relative_rank.nullity
    return row


def write_rank_atlas(
    results: tuple[RankAtlasResult, ...],
    output_dir: Path,
    *,
    stem: str,
) -> tuple[Path, Path]:
    """Write one CSV row and one JSON record per homogeneous case."""
    if not isinstance(results, tuple):
        raise TypeError("results must be a tuple.")
    if not results:
        raise ValueError("results must not be empty.")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path.")
    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a nonempty filename stem.")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    payload = {
        "schema_version": 1,
        "evidence_kind": "numerical_rank_only",
        "basis": "standard_lyndon",
        "bracket_convention": "[V,W](x) = DW(x)V(x) - DV(x)W(x)",
        "matrix_row_order": "evaluation_point_then_state_coordinate",
        "matrix_column_order": "lexicographic_standard_lyndon_basis",
        "nullspace_sign_convention": "largest_absolute_entry_positive",
        "nullspace_basis_note": (
            "sign-canonicalized SVD basis; repeated singular subspaces "
            "may rotate across numerical backends"
        ),
        "dtype": "torch.float64",
        "device": "cpu",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_num_threads": torch.get_num_threads(),
        "torch_config": torch.__config__.show(),
        "result_count": len(results),
        "completed_count": sum(
            result.status == "completed" for result in results
        ),
        "skipped_count": sum(
            result.status == "skipped" for result in results
        ),
        "results": [_result_payload(result) for result in results],
    }
    with json_path.open("w", encoding="utf-8", newline="\n") as json_file:
        json.dump(
            payload,
            json_file,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        json_file.write("\n")

    relative_count = max(
        len(result.case.relative_tolerances)
        for result in results
    )
    relative_fields = tuple(
        field
        for index in range(1, relative_count + 1)
        for field in (
            f"relative_tolerance_{index}",
            f"absolute_tolerance_{index}",
            f"numerical_rank_{index}",
            f"nullity_{index}",
        )
    )
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(*_CSV_BASE_FIELDS, *relative_fields),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(_csv_row(result) for result in results)
    return csv_path, json_path


def _format_number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3e}"


def format_rank_table(results: tuple[RankAtlasResult, ...]) -> str:
    """Return a compact ASCII table with one row per homogeneous case."""
    headers = (
        "architecture",
        "input",
        "state",
        "degree",
        "points",
        "shape",
        "rank",
        "null",
        "sigma_min",
        "tolerance",
        "cond_image",
        "seconds",
        "status",
    )
    rows: list[tuple[str, ...]] = []
    for result in results:
        case = result.case
        diagnostics = result.diagnostics
        if diagnostics is None:
            rank = "-"
            nullity = "-"
            smallest = "-"
            tolerance = "-"
            condition = "-"
        else:
            rank = f"{diagnostics.numerical_rank}/{case.column_count}"
            nullity = str(diagnostics.nullity)
            smallest = _format_number(
                diagnostics.smallest_singular_value
            )
            tolerance = _format_number(diagnostics.standard_tolerance)
            condition = _format_number(
                diagnostics.retained_condition_number
            )
        rows.append(
            (
                case.architecture,
                str(case.input_dim),
                str(case.state_dim),
                str(case.degree),
                str(case.point_count),
                f"{case.row_count}x{case.column_count}",
                rank,
                nullity,
                smallest,
                tolerance,
                condition,
                f"{result.elapsed_seconds:.3f}",
                result.status,
            )
        )

    widths = tuple(
        max(
            len(headers[column]),
            *(len(row[column]) for row in rows),
        )
        for column in range(len(headers))
    )

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(
            value.ljust(widths[index])
            for index, value in enumerate(row)
        )

    separator = "  ".join("-" * width for width in widths)
    return "\n".join(
        (
            format_row(headers),
            separator,
            *(format_row(row) for row in rows),
        )
    )


__all__ = [
    "ARCHITECTURES",
    "ArchitectureName",
    "DEFAULT_DEGREES",
    "DEFAULT_INPUT_DIMS",
    "DEFAULT_MAX_MATRIX_ENTRIES",
    "DEFAULT_POINT_SCALE",
    "DEFAULT_POINT_SEED",
    "DEFAULT_RELATIVE_TOLERANCES",
    "DEFAULT_ROW_OVERSAMPLING",
    "DEFAULT_STATE_DIMS",
    "DEFAULT_WEIGHT_SEED",
    "RankAtlasCase",
    "RankAtlasResult",
    "RankDiagnostics",
    "RelativeRank",
    "architecture_name",
    "build_homogeneous_evaluation_matrix",
    "compute_rank_diagnostics",
    "format_rank_table",
    "full_atlas_cases",
    "make_rank_atlas_case",
    "make_vector_field_config",
    "parameter_scales",
    "run_rank_atlas",
    "run_rank_case",
    "sample_evaluation_points",
    "smoke_atlas_cases",
    "write_rank_atlas",
]
