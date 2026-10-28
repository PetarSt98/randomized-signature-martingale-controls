"""Continuous-time vector fields for the free-Lie rank experiment.

The three supported families are exactly

``shallow biased tanh``
    ``V_i(x) = tanh(A_i x + b_i)``
``shallow bias-free tanh``
    ``V_i(x) = tanh(A_i x)``
``depth-two exponential``
    ``V_i(x) = exp(A_i exp(D_i x))``

Both exponentials and tanh act componentwise. For the reference family,
``D_i`` is stored only through its diagonal entries, never as a dense matrix.
The conservative reference sampling scales are numerical defaults for avoiding
routine overflow; they are not theorem assumptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Callable, TypeAlias

import torch


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_scale(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True)
class ShallowBiasedTanhConfig:
    """Configuration for ``tanh(A_i x + b_i)``."""

    input_dim: int
    state_dim: int
    weight_scale: float = 1.0
    bias_scale: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive_integer("input_dim", self.input_dim)
        _validate_positive_integer("state_dim", self.state_dim)
        _validate_scale("weight_scale", self.weight_scale)
        _validate_scale("bias_scale", self.bias_scale)


@dataclass(frozen=True)
class ShallowBiasFreeTanhConfig:
    """Configuration for ``tanh(A_i x)``."""

    input_dim: int
    state_dim: int
    weight_scale: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive_integer("input_dim", self.input_dim)
        _validate_positive_integer("state_dim", self.state_dim)
        _validate_scale("weight_scale", self.weight_scale)


@dataclass(frozen=True)
class DepthTwoExponentialConfig:
    """Configuration for the reference field ``exp(A_i exp(D_i x))``.

    Sampling gives each outer-matrix entry standard deviation
    ``outer_weight_scale / sqrt(state_dim)`` and each diagonal entry standard
    deviation ``diagonal_scale``. The conservative defaults are numerical
    choices only, not theorem assumptions.
    """

    input_dim: int
    state_dim: int
    outer_weight_scale: float = 0.1
    diagonal_scale: float = 0.1

    def __post_init__(self) -> None:
        _validate_positive_integer("input_dim", self.input_dim)
        _validate_positive_integer("state_dim", self.state_dim)
        _validate_scale("outer_weight_scale", self.outer_weight_scale)
        _validate_scale("diagonal_scale", self.diagonal_scale)


VectorFieldConfig: TypeAlias = (
    ShallowBiasedTanhConfig
    | ShallowBiasFreeTanhConfig
    | DepthTwoExponentialConfig
)


@dataclass(frozen=True, eq=False)
class ShallowBiasedTanhParameters:
    """Matrices and biases for the shallow biased tanh family."""

    weights: torch.Tensor
    biases: torch.Tensor


@dataclass(frozen=True, eq=False)
class ShallowBiasFreeTanhParameters:
    """Matrices for the shallow bias-free tanh family."""

    weights: torch.Tensor


@dataclass(frozen=True, eq=False)
class DepthTwoExponentialParameters:
    """Outer matrices and diagonal entries for the reference family."""

    outer_weights: torch.Tensor
    diagonal_entries: torch.Tensor


Parameters: TypeAlias = (
    ShallowBiasedTanhParameters
    | ShallowBiasFreeTanhParameters
    | DepthTwoExponentialParameters
)
VectorField: TypeAlias = Callable[[torch.Tensor], torch.Tensor]


def _validate_config_type(config: object) -> None:
    if not isinstance(
        config,
        (
            ShallowBiasedTanhConfig,
            ShallowBiasFreeTanhConfig,
            DepthTwoExponentialConfig,
        ),
    ):
        raise TypeError("config must be a supported vector-field configuration.")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be between 0 and 2**63 - 1.")


def _normal(
    shape: tuple[int, ...],
    random_generator: torch.Generator,
    scale: float,
) -> torch.Tensor:
    return torch.randn(
        shape,
        generator=random_generator,
        dtype=torch.float64,
        device="cpu",
    ) * float(scale)


def sample_parameters(config: VectorFieldConfig, seed: int) -> Parameters:
    """Sample float64 parameters with a fresh local CPU generator.

    Shallow weights reproduce the audited finance convention
    ``standard_normal * weight_scale / sqrt(state_dim)``. Biased shallow
    biases use ``standard_normal * bias_scale`` without the square-root factor.
    Equal seeds therefore give identical weight draws for the two shallow
    comparators when their dimensions and weight scales agree.
    """
    _validate_config_type(config)
    _validate_seed(seed)
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(seed)
    matrix_shape = (config.input_dim, config.state_dim, config.state_dim)

    if isinstance(config, ShallowBiasedTanhConfig):
        weights = _normal(
            matrix_shape,
            random_generator,
            float(config.weight_scale) / math.sqrt(config.state_dim),
        )
        biases = _normal(
            (config.input_dim, config.state_dim),
            random_generator,
            float(config.bias_scale),
        )
        return ShallowBiasedTanhParameters(weights=weights, biases=biases)

    if isinstance(config, ShallowBiasFreeTanhConfig):
        weights = _normal(
            matrix_shape,
            random_generator,
            float(config.weight_scale) / math.sqrt(config.state_dim),
        )
        return ShallowBiasFreeTanhParameters(weights=weights)

    outer_weights = _normal(
        matrix_shape,
        random_generator,
        float(config.outer_weight_scale) / math.sqrt(config.state_dim),
    )
    diagonal_entries = _normal(
        (config.input_dim, config.state_dim),
        random_generator,
        float(config.diagonal_scale),
    )
    return DepthTwoExponentialParameters(
        outer_weights=outer_weights,
        diagonal_entries=diagonal_entries,
    )


def _validate_tensor(
    name: str,
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}."
        )
    if tensor.dtype != torch.float64:
        raise TypeError(f"{name} must have dtype torch.float64.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")


def _validate_parameters(
    config: VectorFieldConfig,
    parameters: Parameters,
) -> torch.device:
    _validate_config_type(config)
    matrix_shape = (config.input_dim, config.state_dim, config.state_dim)
    vector_shape = (config.input_dim, config.state_dim)

    if isinstance(config, ShallowBiasedTanhConfig):
        if not isinstance(parameters, ShallowBiasedTanhParameters):
            raise TypeError(
                "ShallowBiasedTanhConfig requires ShallowBiasedTanhParameters."
            )
        _validate_tensor("weights", parameters.weights, matrix_shape)
        _validate_tensor("biases", parameters.biases, vector_shape)
        if parameters.weights.device != parameters.biases.device:
            raise ValueError("weights and biases must be on the same device.")
        return parameters.weights.device

    if isinstance(config, ShallowBiasFreeTanhConfig):
        if not isinstance(parameters, ShallowBiasFreeTanhParameters):
            raise TypeError(
                "ShallowBiasFreeTanhConfig requires ShallowBiasFreeTanhParameters."
            )
        _validate_tensor("weights", parameters.weights, matrix_shape)
        return parameters.weights.device

    if isinstance(config, DepthTwoExponentialConfig):
        if not isinstance(parameters, DepthTwoExponentialParameters):
            raise TypeError(
                "DepthTwoExponentialConfig requires DepthTwoExponentialParameters."
            )
        _validate_tensor("outer_weights", parameters.outer_weights, matrix_shape)
        _validate_tensor(
            "diagonal_entries",
            parameters.diagonal_entries,
            vector_shape,
        )
        if parameters.outer_weights.device != parameters.diagonal_entries.device:
            raise ValueError(
                "outer_weights and diagonal_entries must be on the same device."
            )
        return parameters.outer_weights.device

    raise AssertionError("unreachable supported configuration branch.")


def _validate_field_index(field_index: int, input_dim: int) -> None:
    if isinstance(field_index, bool) or not isinstance(field_index, int):
        raise TypeError("field_index must be an integer.")
    if not 0 <= field_index < input_dim:
        raise ValueError(f"field_index must be in [0, {input_dim - 1}].")


def _validate_state(
    x: torch.Tensor,
    state_dim: int,
    parameter_device: torch.device,
) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor.")
    if tuple(x.shape) != (state_dim,):
        raise ValueError(f"x must have shape ({state_dim},), got {tuple(x.shape)}.")
    if x.dtype != torch.float64:
        raise TypeError("x must have dtype torch.float64.")
    if x.device != parameter_device:
        raise ValueError(
            f"x must be on device {parameter_device}, got device {x.device}."
        )
    if not torch.isfinite(x).all():
        raise ValueError("x must contain only finite values.")


def _validate_output(output: torch.Tensor, state_dim: int) -> None:
    if tuple(output.shape) != (state_dim,):
        raise RuntimeError(
            f"vector field returned shape {tuple(output.shape)}, "
            f"expected ({state_dim},)."
        )
    if output.dtype != torch.float64:
        raise RuntimeError("vector field output must have dtype torch.float64.")
    if not torch.isfinite(output).all():
        raise FloatingPointError("vector field output contains a non-finite value.")


def _build_vector_field(
    config: VectorFieldConfig,
    parameters: Parameters,
    field_index: int,
    parameter_device: torch.device,
) -> VectorField:
    if isinstance(config, ShallowBiasedTanhConfig):
        if not isinstance(parameters, ShallowBiasedTanhParameters):
            raise TypeError("incompatible config and parameters.")
        weights = parameters.weights[field_index]
        biases = parameters.biases[field_index]

        def vector_field(x: torch.Tensor) -> torch.Tensor:
            _validate_state(x, config.state_dim, parameter_device)
            output = torch.tanh(weights @ x + biases)
            _validate_output(output, config.state_dim)
            return output

        return vector_field

    if isinstance(config, ShallowBiasFreeTanhConfig):
        if not isinstance(parameters, ShallowBiasFreeTanhParameters):
            raise TypeError("incompatible config and parameters.")
        weights = parameters.weights[field_index]

        def vector_field(x: torch.Tensor) -> torch.Tensor:
            _validate_state(x, config.state_dim, parameter_device)
            output = torch.tanh(weights @ x)
            _validate_output(output, config.state_dim)
            return output

        return vector_field

    if not isinstance(parameters, DepthTwoExponentialParameters):
        raise TypeError("incompatible config and parameters.")
    outer_weights = parameters.outer_weights[field_index]
    diagonal_entries = parameters.diagonal_entries[field_index]

    def vector_field(x: torch.Tensor) -> torch.Tensor:
        _validate_state(x, config.state_dim, parameter_device)
        inner = torch.exp(diagonal_entries * x)
        output = torch.exp(outer_weights @ inner)
        _validate_output(output, config.state_dim)
        return output

    return vector_field


def make_vector_field(
    config: VectorFieldConfig,
    parameters: Parameters,
    field_index: int,
) -> VectorField:
    """Build one field while validating its zero-based ``field_index``."""
    parameter_device = _validate_parameters(config, parameters)
    _validate_field_index(field_index, config.input_dim)
    return _build_vector_field(
        config=config,
        parameters=parameters,
        field_index=field_index,
        parameter_device=parameter_device,
    )


def make_vector_fields(
    config: VectorFieldConfig,
    parameters: Parameters,
) -> tuple[VectorField, ...]:
    """Build all ``input_dim`` fields from one shared parameter container."""
    parameter_device = _validate_parameters(config, parameters)
    return tuple(
        _build_vector_field(
            config=config,
            parameters=parameters,
            field_index=field_index,
            parameter_device=parameter_device,
        )
        for field_index in range(config.input_dim)
    )


__all__ = [
    "DepthTwoExponentialConfig",
    "DepthTwoExponentialParameters",
    "Parameters",
    "ShallowBiasedTanhConfig",
    "ShallowBiasedTanhParameters",
    "ShallowBiasFreeTanhConfig",
    "ShallowBiasFreeTanhParameters",
    "VectorField",
    "VectorFieldConfig",
    "make_vector_field",
    "make_vector_fields",
    "sample_parameters",
]
