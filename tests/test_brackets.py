"""Tests for recursive represented Lie-bracket evaluation.

Runs with pytest (``pytest tests``) or as a script
(``python tests/test_brackets.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lie_reservoir import (  # noqa: E402
    Bracket,
    DepthTwoExponentialConfig,
    Generator,
    LieNode,
    Parameters,
    ShallowBiasedTanhConfig,
    ShallowBiasedTanhParameters,
    ShallowBiasFreeTanhConfig,
    ShallowBiasFreeTanhParameters,
    VectorField,
    VectorFieldConfig,
    evaluate_lie_node,
    lyndon_basis,
    make_lie_evaluator,
    make_vector_fields,
    sample_parameters,
)


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


def _architecture_cases(
    input_dim: int,
    state_dim: int,
) -> tuple[tuple[str, VectorFieldConfig, Parameters], ...]:
    configs: tuple[VectorFieldConfig, ...] = (
        ShallowBiasedTanhConfig(input_dim=input_dim, state_dim=state_dim),
        ShallowBiasFreeTanhConfig(input_dim=input_dim, state_dim=state_dim),
        DepthTwoExponentialConfig(input_dim=input_dim, state_dim=state_dim),
    )
    return tuple(
        (
            type(config).__name__,
            config,
            sample_parameters(config, seed=73 + offset),
        )
        for offset, config in enumerate(configs)
    )


def _parameter_tensors(parameters: Parameters) -> tuple[torch.Tensor, ...]:
    if isinstance(parameters, ShallowBiasedTanhParameters):
        return parameters.weights, parameters.biases
    if isinstance(parameters, ShallowBiasFreeTanhParameters):
        return (parameters.weights,)
    return parameters.outer_weights, parameters.diagonal_entries


def _represented_linear_matrix(
    node: LieNode,
    matrices: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if isinstance(node, Generator):
        return matrices[node.index]
    left = _represented_linear_matrix(node.left, matrices)
    right = _represented_linear_matrix(node.right, matrices)
    return right @ left - left @ right


def _central_directional_derivative(
    vector_field: VectorField,
    x: torch.Tensor,
    direction: torch.Tensor,
    *,
    step: float,
) -> torch.Tensor:
    return (
        vector_field(x + step * direction)
        - vector_field(x - step * direction)
    ) / (2.0 * step)


def test_constant_fields_have_zero_bracket() -> None:
    first_value = torch.tensor([1.5, -2.0], dtype=torch.float64)
    second_value = torch.tensor([-0.5, 3.0], dtype=torch.float64)

    def first(_x: torch.Tensor) -> torch.Tensor:
        return first_value

    def second(_x: torch.Tensor) -> torch.Tensor:
        return second_value

    vector_fields: tuple[VectorField, ...] = (first, second)
    x = torch.tensor([0.25, -0.75], dtype=torch.float64)
    assert torch.equal(
        evaluate_lie_node(Generator(0), vector_fields, x),
        first_value,
    )
    bracket = evaluate_lie_node(
        Bracket(Generator(0), Generator(1)),
        vector_fields,
        x,
    )
    assert torch.equal(bracket, torch.zeros_like(x))


def test_linear_fields_fix_the_bracket_sign() -> None:
    first_matrix = torch.tensor(
        [[1.0, 2.0], [0.0, -1.0]],
        dtype=torch.float64,
    )
    second_matrix = torch.tensor(
        [[0.0, 1.0], [3.0, 2.0]],
        dtype=torch.float64,
    )
    x = torch.tensor([2.0, -1.0], dtype=torch.float64)
    vector_fields: tuple[VectorField, ...] = (
        lambda state: first_matrix @ state,
        lambda state: second_matrix @ state,
    )
    actual = evaluate_lie_node(
        Bracket(Generator(0), Generator(1)),
        vector_fields,
        x,
    )
    expected = (second_matrix @ first_matrix - first_matrix @ second_matrix) @ x
    assert torch.equal(expected, torch.tensor([-6.0, 6.0], dtype=torch.float64))
    assert torch.equal(actual, expected)


def test_scalar_nonlinear_bracket_uses_jacobians_at_x() -> None:
    def first(x: torch.Tensor) -> torch.Tensor:
        return x**2 + 1.0

    def second(x: torch.Tensor) -> torch.Tensor:
        return x**3 - 2.0 * x

    actual = evaluate_lie_node(
        Bracket(Generator(0), Generator(1)),
        (first, second),
        torch.tensor([2.0], dtype=torch.float64),
    )
    assert torch.equal(actual, torch.tensor([34.0], dtype=torch.float64))


def test_nonlinear_architecture_brackets_match_finite_differences() -> None:
    x = torch.tensor([0.2, -0.3], dtype=torch.float64)
    node = Bracket(Generator(0), Generator(1))
    step = 1e-5

    for _name, config, parameters in _architecture_cases(2, 2):
        vector_fields = make_vector_fields(config, parameters)
        first_value = vector_fields[0](x)
        second_value = vector_fields[1](x)
        finite_difference = _central_directional_derivative(
            vector_fields[1],
            x,
            first_value,
            step=step,
        ) - _central_directional_derivative(
            vector_fields[0],
            x,
            second_value,
            step=step,
        )
        actual = evaluate_lie_node(node, vector_fields, x)
        torch.testing.assert_close(
            actual,
            finite_difference,
            rtol=1e-8,
            atol=1e-10,
        )


def test_all_lyndon_nodes_match_exact_linear_matrix_recursion() -> None:
    matrices = (
        torch.tensor(
            [[1.0, 2.0, 0.0], [0.0, -1.0, 1.0], [1.0, 0.0, 2.0]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[0.0, 1.0, 1.0], [2.0, 0.0, -1.0], [0.0, 1.0, 1.0]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[1.0, 1.0, -2.0], [0.0, 2.0, 1.0], [1.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
    )
    vector_fields = tuple(
        (lambda x, matrix=matrix: matrix @ x)
        for matrix in matrices
    )
    x = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)

    for degree in range(1, 5):
        for node in lyndon_basis(input_dim=3, degree=degree):
            actual = evaluate_lie_node(node, vector_fields, x)
            expected = _represented_linear_matrix(node, matrices) @ x
            assert torch.equal(actual, expected)


def test_antisymmetry_and_jacobi_for_all_architectures() -> None:
    x = torch.tensor([0.1, -0.15, 0.05], dtype=torch.float64)
    first = Generator(0)
    second = Generator(1)
    third = Generator(2)
    first_second = Bracket(first, second)
    second_first = Bracket(second, first)
    jacobi_nodes = (
        Bracket(first, Bracket(second, third)),
        Bracket(second, Bracket(third, first)),
        Bracket(third, Bracket(first, second)),
    )

    for _name, config, parameters in _architecture_cases(3, 3):
        vector_fields = make_vector_fields(config, parameters)
        forward = evaluate_lie_node(first_second, vector_fields, x)
        reverse = evaluate_lie_node(second_first, vector_fields, x)
        torch.testing.assert_close(
            forward,
            -reverse,
            rtol=1e-12,
            atol=1e-13,
        )

        jacobi_terms = tuple(
            evaluate_lie_node(node, vector_fields, x)
            for node in jacobi_nodes
        )
        jacobi_sum = jacobi_terms[0] + jacobi_terms[1] + jacobi_terms[2]
        # Nested reverse-mode operations can differ in rounding order. This
        # tolerance is below 10,000 float64 eps at the tested unit scale.
        torch.testing.assert_close(
            jacobi_sum,
            torch.zeros_like(x),
            rtol=0.0,
            atol=2e-12,
        )


def test_architectures_evaluate_every_basis_node_through_degree_four() -> None:
    x = torch.tensor([0.1, -0.15, 0.05], dtype=torch.float64)
    for _name, config, parameters in _architecture_cases(3, 3):
        vector_fields = make_vector_fields(config, parameters)
        for degree in range(1, 5):
            for node in lyndon_basis(input_dim=3, degree=degree):
                value = evaluate_lie_node(node, vector_fields, x)
                assert value.shape == (3,)
                assert value.dtype == torch.float64
                assert torch.isfinite(value).all()
                if isinstance(node, Generator):
                    assert torch.equal(value, vector_fields[node.index](x))


def test_degree_four_preserves_state_and_parameter_graphs() -> None:
    degree_four_basis = lyndon_basis(input_dim=3, degree=4)
    balanced_node = Bracket(
        Bracket(Generator(0), Generator(1)),
        Bracket(Generator(0), Generator(2)),
    )
    assert balanced_node in degree_four_basis
    graph_nodes = (degree_four_basis[0], balanced_node)

    for _name, config, parameters in _architecture_cases(3, 3):
        parameter_tensors = _parameter_tensors(parameters)
        for tensor in parameter_tensors:
            tensor.requires_grad_(True)
        vector_fields = make_vector_fields(config, parameters)
        for node in graph_nodes:
            evaluator = make_lie_evaluator(node, vector_fields)
            x = torch.tensor(
                [0.1, -0.15, 0.05],
                dtype=torch.float64,
                requires_grad=True,
            )

            value = evaluator(x)
            jacobian = torch.func.jacrev(evaluator)(x)
            value_gradients = torch.autograd.grad(
                value.sum(),
                (x, *parameter_tensors),
                create_graph=True,
            )
            jacobian_gradient = torch.autograd.grad(jacobian.sum(), x)[0]

            assert value.grad_fn is not None
            assert jacobian.grad_fn is not None
            assert torch.isfinite(value).all()
            assert torch.isfinite(jacobian).all()
            assert torch.isfinite(jacobian_gradient).all()
            assert all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in value_gradients
            )


def test_evaluator_validation() -> None:
    identity: VectorField = lambda x: x
    valid_fields = (identity,)
    _assert_raises(TypeError, make_lie_evaluator, object(), valid_fields)
    _assert_raises(TypeError, make_lie_evaluator, Generator(0), [identity])
    _assert_raises(ValueError, make_lie_evaluator, Generator(0), ())
    _assert_raises(TypeError, make_lie_evaluator, Generator(0), (object(),))
    _assert_raises(
        ValueError,
        make_lie_evaluator,
        Generator(1),
        valid_fields,
    )

    evaluator = make_lie_evaluator(Generator(0), valid_fields)
    _assert_raises(TypeError, evaluator, [1.0])
    _assert_raises(
        ValueError,
        evaluator,
        torch.ones((1, 1), dtype=torch.float64),
    )
    _assert_raises(TypeError, evaluator, torch.ones(1, dtype=torch.float32))
    _assert_raises(
        ValueError,
        evaluator,
        torch.tensor([float("nan")], dtype=torch.float64),
    )

    wrong_shape: VectorField = lambda x: torch.ones(
        x.numel() + 1,
        dtype=torch.float64,
        device=x.device,
    )
    wrong_dtype: VectorField = lambda x: torch.ones(
        x.shape,
        dtype=torch.float32,
        device=x.device,
    )
    nonfinite: VectorField = lambda x: torch.full_like(x, float("inf"))
    x = torch.ones(1, dtype=torch.float64)
    _assert_raises(
        ValueError,
        make_lie_evaluator(Generator(0), (wrong_shape,)),
        x,
    )
    _assert_raises(
        TypeError,
        make_lie_evaluator(Generator(0), (wrong_dtype,)),
        x,
    )
    _assert_raises(
        FloatingPointError,
        make_lie_evaluator(Generator(0), (nonfinite,)),
        x,
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
