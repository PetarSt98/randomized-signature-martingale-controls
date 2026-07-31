"""Tests for the three continuous-time vector-field families.

Runs with pytest (``pytest tests``) or as a script
(``python tests/test_vector_fields.py``).
"""

from __future__ import annotations

import math
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lie_reservoir import (  # noqa: E402
    DepthTwoExponentialConfig,
    DepthTwoExponentialParameters,
    Parameters,
    ShallowBiasedTanhConfig,
    ShallowBiasedTanhParameters,
    ShallowBiasFreeTanhConfig,
    ShallowBiasFreeTanhParameters,
    make_vector_field,
    make_vector_fields,
    sample_parameters,
)


def _parameter_tensors(parameters: Parameters) -> tuple[torch.Tensor, ...]:
    if isinstance(parameters, ShallowBiasedTanhParameters):
        return parameters.weights, parameters.biases
    if isinstance(parameters, ShallowBiasFreeTanhParameters):
        return (parameters.weights,)
    return parameters.outer_weights, parameters.diagonal_entries


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


def test_sampled_shapes_dtype_and_structural_diagonal() -> None:
    biased = sample_parameters(
        ShallowBiasedTanhConfig(input_dim=3, state_dim=2),
        seed=1,
    )
    bias_free = sample_parameters(
        ShallowBiasFreeTanhConfig(input_dim=3, state_dim=2),
        seed=1,
    )
    reference = sample_parameters(
        DepthTwoExponentialConfig(input_dim=3, state_dim=2),
        seed=1,
    )
    assert isinstance(biased, ShallowBiasedTanhParameters)
    assert isinstance(bias_free, ShallowBiasFreeTanhParameters)
    assert isinstance(reference, DepthTwoExponentialParameters)
    assert biased.weights.shape == (3, 2, 2)
    assert biased.biases.shape == (3, 2)
    assert bias_free.weights.shape == (3, 2, 2)
    assert reference.outer_weights.shape == (3, 2, 2)
    assert reference.diagonal_entries.shape == (3, 2)
    assert reference.diagonal_entries.ndim == 2
    assert reference.diagonal_entries.numel() == 6
    assert tuple(field.name for field in fields(reference)) == (
        "outer_weights",
        "diagonal_entries",
    )
    assert all(
        tensor.dtype == torch.float64
        for parameters in (biased, bias_free, reference)
        for tensor in _parameter_tensors(parameters)
    )
    _assert_raises(
        FrozenInstanceError,
        setattr,
        reference,
        "diagonal_entries",
        torch.zeros((3, 2), dtype=torch.float64),
    )


def test_field_count_output_shape_and_finiteness() -> None:
    configs = (
        ShallowBiasedTanhConfig(input_dim=3, state_dim=2),
        ShallowBiasFreeTanhConfig(input_dim=3, state_dim=2),
        DepthTwoExponentialConfig(input_dim=3, state_dim=2),
    )
    x = torch.tensor([0.1, -0.2], dtype=torch.float64)
    for seed, config in enumerate(configs, start=2):
        parameters = sample_parameters(config, seed=seed)
        vector_fields = make_vector_fields(config, parameters)
        assert isinstance(vector_fields, tuple)
        assert len(vector_fields) == config.input_dim
        for vector_field in vector_fields:
            output = vector_field(x)
            assert output.shape == (config.state_dim,)
            assert output.dtype == torch.float64
            assert torch.isfinite(output).all()


def test_sampling_is_local_and_deterministic() -> None:
    configs = (
        ShallowBiasedTanhConfig(input_dim=2, state_dim=3),
        ShallowBiasFreeTanhConfig(input_dim=2, state_dim=3),
        DepthTwoExponentialConfig(input_dim=2, state_dim=3),
    )
    global_random_state = torch.random.get_rng_state().clone()
    default_dtype = torch.get_default_dtype()
    for config in configs:
        first = sample_parameters(config, seed=7)
        repeated = sample_parameters(config, seed=7)
        different = sample_parameters(config, seed=8)
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                _parameter_tensors(first),
                _parameter_tensors(repeated),
                strict=True,
            )
        )
        assert any(
            not torch.equal(left, right)
            for left, right in zip(
                _parameter_tensors(first),
                _parameter_tensors(different),
                strict=True,
            )
        )
    assert torch.equal(global_random_state, torch.random.get_rng_state())
    assert torch.get_default_dtype() == default_dtype

    biased = sample_parameters(
        ShallowBiasedTanhConfig(input_dim=2, state_dim=3, weight_scale=0.7),
        seed=9,
    )
    bias_free = sample_parameters(
        ShallowBiasFreeTanhConfig(input_dim=2, state_dim=3, weight_scale=0.7),
        seed=9,
    )
    assert isinstance(biased, ShallowBiasedTanhParameters)
    assert isinstance(bias_free, ShallowBiasFreeTanhParameters)
    assert torch.equal(biased.weights, bias_free.weights)


def test_shallow_sampling_matches_audited_scaling() -> None:
    config = ShallowBiasedTanhConfig(
        input_dim=2,
        state_dim=3,
        weight_scale=1.7,
        bias_scale=0.4,
    )
    actual = sample_parameters(config, seed=19)
    assert isinstance(actual, ShallowBiasedTanhParameters)

    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(19)
    expected_weights = torch.randn(
        (2, 3, 3),
        generator=random_generator,
        dtype=torch.float64,
    ) * (1.7 / math.sqrt(3))
    expected_biases = torch.randn(
        (2, 3),
        generator=random_generator,
        dtype=torch.float64,
    ) * 0.4
    assert torch.equal(actual.weights, expected_weights)
    assert torch.equal(actual.biases, expected_biases)


def test_reference_sampling_scales_are_explicit_and_honoured() -> None:
    config = DepthTwoExponentialConfig(
        input_dim=2,
        state_dim=3,
        outer_weight_scale=0.2,
        diagonal_scale=0.3,
    )
    actual = sample_parameters(config, seed=23)
    assert isinstance(actual, DepthTwoExponentialParameters)

    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(23)
    expected_outer = torch.randn(
        (2, 3, 3),
        generator=random_generator,
        dtype=torch.float64,
    ) * (0.2 / math.sqrt(3))
    expected_diagonal = torch.randn(
        (2, 3),
        generator=random_generator,
        dtype=torch.float64,
    ) * 0.3
    assert torch.equal(actual.outer_weights, expected_outer)
    assert torch.equal(actual.diagonal_entries, expected_diagonal)

    zero = sample_parameters(
        DepthTwoExponentialConfig(
            input_dim=1,
            state_dim=2,
            outer_weight_scale=0.0,
            diagonal_scale=0.0,
        ),
        seed=23,
    )
    assert isinstance(zero, DepthTwoExponentialParameters)
    assert torch.count_nonzero(zero.outer_weights) == 0
    assert torch.count_nonzero(zero.diagonal_entries) == 0


def test_manual_one_dimensional_formulas() -> None:
    x = torch.tensor([0.4], dtype=torch.float64)

    biased_config = ShallowBiasedTanhConfig(input_dim=1, state_dim=1)
    biased_parameters = ShallowBiasedTanhParameters(
        weights=torch.tensor([[[2.0]]], dtype=torch.float64),
        biases=torch.tensor([[-0.25]], dtype=torch.float64),
    )
    biased = make_vector_fields(biased_config, biased_parameters)[0](x)
    expected_biased = torch.tanh(torch.tensor([0.55], dtype=torch.float64))
    torch.testing.assert_close(biased, expected_biased, rtol=1e-15, atol=1e-15)

    bias_free_config = ShallowBiasFreeTanhConfig(input_dim=1, state_dim=1)
    bias_free_parameters = ShallowBiasFreeTanhParameters(
        weights=torch.tensor([[[2.0]]], dtype=torch.float64),
    )
    bias_free = make_vector_fields(bias_free_config, bias_free_parameters)[0](x)
    expected_bias_free = torch.tanh(torch.tensor([0.8], dtype=torch.float64))
    torch.testing.assert_close(
        bias_free,
        expected_bias_free,
        rtol=1e-15,
        atol=1e-15,
    )

    reference_config = DepthTwoExponentialConfig(input_dim=1, state_dim=1)
    reference_parameters = DepthTwoExponentialParameters(
        outer_weights=torch.tensor([[[0.5]]], dtype=torch.float64),
        diagonal_entries=torch.tensor([[-0.75]], dtype=torch.float64),
    )
    reference = make_vector_fields(reference_config, reference_parameters)[0](x)
    expected_reference = torch.exp(
        torch.tensor([0.5], dtype=torch.float64)
        * torch.exp(torch.tensor([-0.3], dtype=torch.float64))
    )
    torch.testing.assert_close(
        reference,
        expected_reference,
        rtol=1e-15,
        atol=1e-15,
    )


def test_nonsymmetric_two_dimensional_formulas() -> None:
    x = torch.tensor([0.3, -0.2], dtype=torch.float64)
    weights = torch.tensor(
        [[[1.0, 2.0], [-0.5, 0.25]]],
        dtype=torch.float64,
    )
    biases = torch.tensor([[0.1, -0.3]], dtype=torch.float64)
    diagonal_entries = torch.tensor([[0.4, -0.7]], dtype=torch.float64)

    biased = make_vector_fields(
        ShallowBiasedTanhConfig(input_dim=1, state_dim=2),
        ShallowBiasedTanhParameters(weights=weights, biases=biases),
    )[0]
    bias_free = make_vector_fields(
        ShallowBiasFreeTanhConfig(input_dim=1, state_dim=2),
        ShallowBiasFreeTanhParameters(weights=weights),
    )[0]
    reference = make_vector_fields(
        DepthTwoExponentialConfig(input_dim=1, state_dim=2),
        DepthTwoExponentialParameters(
            outer_weights=weights,
            diagonal_entries=diagonal_entries,
        ),
    )[0]

    torch.testing.assert_close(
        biased(x),
        torch.tanh(weights[0] @ x + biases[0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        bias_free(x),
        torch.tanh(weights[0] @ x),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        reference(x),
        torch.exp(weights[0] @ torch.exp(diagonal_entries[0] * x)),
        rtol=0.0,
        atol=0.0,
    )


def test_each_field_uses_its_own_parameter_slice() -> None:
    x = torch.tensor([0.2, -0.4], dtype=torch.float64)
    weights = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, 0.5], [0.25, -2.0]],
        ],
        dtype=torch.float64,
    )
    biases = torch.tensor(
        [[0.1, -0.2], [0.3, 0.4]],
        dtype=torch.float64,
    )
    diagonal_entries = torch.tensor(
        [[0.2, -0.3], [-0.4, 0.5]],
        dtype=torch.float64,
    )
    cases = (
        (
            ShallowBiasedTanhConfig(input_dim=2, state_dim=2),
            ShallowBiasedTanhParameters(weights=weights, biases=biases),
            tuple(torch.tanh(weights[index] @ x + biases[index]) for index in range(2)),
        ),
        (
            ShallowBiasFreeTanhConfig(input_dim=2, state_dim=2),
            ShallowBiasFreeTanhParameters(weights=weights),
            tuple(torch.tanh(weights[index] @ x) for index in range(2)),
        ),
        (
            DepthTwoExponentialConfig(input_dim=2, state_dim=2),
            DepthTwoExponentialParameters(
                outer_weights=weights,
                diagonal_entries=diagonal_entries,
            ),
            tuple(
                torch.exp(
                    weights[index]
                    @ torch.exp(diagonal_entries[index] * x)
                )
                for index in range(2)
            ),
        ),
    )

    for config, parameters, expected_outputs in cases:
        all_fields = make_vector_fields(config, parameters)
        for field_index, expected in enumerate(expected_outputs):
            torch.testing.assert_close(
                all_fields[field_index](x),
                expected,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                make_vector_field(config, parameters, field_index)(x),
                expected,
                rtol=0.0,
                atol=0.0,
            )


def test_zero_bias_matches_bias_free_exactly() -> None:
    bias_free_config = ShallowBiasFreeTanhConfig(input_dim=2, state_dim=3)
    bias_free_parameters = sample_parameters(bias_free_config, seed=29)
    assert isinstance(bias_free_parameters, ShallowBiasFreeTanhParameters)
    biased_config = ShallowBiasedTanhConfig(input_dim=2, state_dim=3)
    biased_parameters = ShallowBiasedTanhParameters(
        weights=bias_free_parameters.weights,
        biases=torch.zeros((2, 3), dtype=torch.float64),
    )
    biased_fields = make_vector_fields(biased_config, biased_parameters)
    bias_free_fields = make_vector_fields(bias_free_config, bias_free_parameters)
    states = (
        torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
        torch.tensor([-0.4, 0.0, 0.2], dtype=torch.float64),
    )
    for x in states:
        for biased, bias_free in zip(
            biased_fields,
            bias_free_fields,
            strict=True,
        ):
            assert torch.equal(biased(x), bias_free(x))


def test_jacobians_and_parameter_graphs_are_preserved() -> None:
    weights = torch.tensor(
        [[[0.6, -0.2], [0.3, 0.5]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    biases = torch.tensor(
        [[0.1, -0.15]],
        dtype=torch.float64,
        requires_grad=True,
    )
    x = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    biased = make_vector_fields(
        ShallowBiasedTanhConfig(input_dim=1, state_dim=2),
        ShallowBiasedTanhParameters(weights=weights, biases=biases),
    )[0]
    output = biased(x)
    jacobian = torch.autograd.functional.jacobian(
        biased,
        x,
        create_graph=True,
    )
    expected_jacobian = (1.0 - output**2)[:, None] * weights[0]
    torch.testing.assert_close(jacobian, expected_jacobian)
    gradients = torch.autograd.grad(
        output.sum(),
        (x, weights, biases),
        create_graph=True,
    )
    assert output.requires_grad and output.grad_fn is not None
    assert all(gradient is not None for gradient in gradients)

    outer_weights = torch.tensor(
        [[[0.2, -0.1], [0.4, 0.3]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    diagonal_entries = torch.tensor(
        [[0.25, -0.35]],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference_x = torch.tensor(
        [0.15, -0.2],
        dtype=torch.float64,
        requires_grad=True,
    )
    reference = make_vector_fields(
        DepthTwoExponentialConfig(input_dim=1, state_dim=2),
        DepthTwoExponentialParameters(
            outer_weights=outer_weights,
            diagonal_entries=diagonal_entries,
        ),
    )[0]
    reference_output = reference(reference_x)
    reference_jacobian = torch.autograd.functional.jacobian(
        reference,
        reference_x,
        create_graph=True,
    )
    inner = torch.exp(diagonal_entries[0] * reference_x)
    expected_reference_jacobian = (
        reference_output[:, None]
        * outer_weights[0]
        * (inner * diagonal_entries[0])[None, :]
    )
    torch.testing.assert_close(
        reference_jacobian,
        expected_reference_jacobian,
    )
    reference_gradients = torch.autograd.grad(
        reference_output.sum(),
        (reference_x, outer_weights, diagonal_entries),
        create_graph=True,
    )
    assert reference_output.requires_grad
    assert all(gradient is not None for gradient in reference_gradients)


def test_default_scale_smoke_values_and_jacobians_are_finite() -> None:
    configs = (
        ShallowBiasedTanhConfig(input_dim=2, state_dim=3),
        ShallowBiasFreeTanhConfig(input_dim=2, state_dim=3),
        DepthTwoExponentialConfig(input_dim=2, state_dim=3),
    )
    x = torch.linspace(-0.2, 0.2, 3, dtype=torch.float64)
    for seed, config in enumerate(configs, start=31):
        vector_fields = make_vector_fields(
            config,
            sample_parameters(config, seed=seed),
        )
        for vector_field in vector_fields:
            output = vector_field(x)
            jacobian = torch.autograd.functional.jacobian(vector_field, x)
            assert torch.isfinite(output).all()
            assert torch.isfinite(jacobian).all()


def test_configuration_parameter_and_field_index_validation() -> None:
    _assert_raises(TypeError, sample_parameters, object(), 0)
    _assert_raises(TypeError, make_vector_fields, object(), object())
    _assert_raises(TypeError, ShallowBiasedTanhConfig, True, 2)
    _assert_raises(ValueError, ShallowBiasFreeTanhConfig, 2, 0)
    _assert_raises(
        ValueError,
        DepthTwoExponentialConfig,
        2,
        2,
        -0.1,
    )
    _assert_raises(TypeError, sample_parameters, ShallowBiasFreeTanhConfig(1, 1), True)
    _assert_raises(ValueError, sample_parameters, ShallowBiasFreeTanhConfig(1, 1), -1)

    biased_config = ShallowBiasedTanhConfig(input_dim=2, state_dim=2)
    biased_parameters = sample_parameters(biased_config, seed=37)
    assert isinstance(biased_parameters, ShallowBiasedTanhParameters)
    _assert_raises(
        TypeError,
        make_vector_fields,
        biased_config,
        ShallowBiasFreeTanhParameters(weights=biased_parameters.weights),
    )
    _assert_raises(
        TypeError,
        make_vector_field,
        biased_config,
        biased_parameters,
        True,
    )
    _assert_raises(
        ValueError,
        make_vector_field,
        biased_config,
        biased_parameters,
        -1,
    )
    _assert_raises(
        ValueError,
        make_vector_field,
        biased_config,
        biased_parameters,
        2,
    )
    assert callable(make_vector_field(biased_config, biased_parameters, 0))
    assert callable(make_vector_field(biased_config, biased_parameters, 1))


def test_parameter_state_and_finite_output_validation() -> None:
    config = ShallowBiasedTanhConfig(input_dim=2, state_dim=2)
    valid = sample_parameters(config, seed=41)
    assert isinstance(valid, ShallowBiasedTanhParameters)
    _assert_raises(
        ValueError,
        make_vector_fields,
        config,
        ShallowBiasedTanhParameters(
            weights=torch.zeros((1, 2, 2), dtype=torch.float64),
            biases=torch.zeros((2, 2), dtype=torch.float64),
        ),
    )
    _assert_raises(
        ValueError,
        make_vector_fields,
        config,
        ShallowBiasedTanhParameters(
            weights=torch.zeros((2, 2, 2), dtype=torch.float64),
            biases=torch.zeros((2, 3), dtype=torch.float64),
        ),
    )
    _assert_raises(
        TypeError,
        make_vector_fields,
        config,
        ShallowBiasedTanhParameters(
            weights=torch.zeros((2, 2, 2), dtype=torch.float32),
            biases=torch.zeros((2, 2), dtype=torch.float64),
        ),
    )

    reference_config = DepthTwoExponentialConfig(input_dim=2, state_dim=2)
    _assert_raises(
        ValueError,
        make_vector_fields,
        reference_config,
        DepthTwoExponentialParameters(
            outer_weights=torch.zeros((2, 2, 2), dtype=torch.float64),
            diagonal_entries=torch.zeros((2, 2, 2), dtype=torch.float64),
        ),
    )

    vector_field = make_vector_fields(config, valid)[0]
    _assert_raises(TypeError, vector_field, [0.0, 0.0])
    _assert_raises(
        ValueError,
        vector_field,
        torch.zeros((2, 1), dtype=torch.float64),
    )
    _assert_raises(TypeError, vector_field, torch.zeros(2, dtype=torch.float32))
    _assert_raises(
        ValueError,
        vector_field,
        torch.tensor([float("nan"), 0.0], dtype=torch.float64),
    )

    overflowing = make_vector_fields(
        DepthTwoExponentialConfig(input_dim=1, state_dim=1),
        DepthTwoExponentialParameters(
            outer_weights=torch.tensor([[[1_000.0]]], dtype=torch.float64),
            diagonal_entries=torch.zeros((1, 1), dtype=torch.float64),
        ),
    )[0]
    _assert_raises(
        FloatingPointError,
        overflowing,
        torch.zeros(1, dtype=torch.float64),
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
