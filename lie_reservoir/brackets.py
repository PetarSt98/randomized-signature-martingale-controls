"""Recursive evaluation of represented free-Lie brackets.

For vector fields ``V`` and ``W``, this module uses exactly

``[V, W](x) = DW(x) V(x) - DV(x) W(x)``.

The recursive evaluator is constructed once from an immutable :class:`LieNode`.
Only the evaluator structure is retained; values and Jacobians are recomputed
at every state, so no state- or parameter-dependent graph is cached.
"""

from __future__ import annotations

import torch

from .free_lie import Bracket, Generator, LieNode
from .vector_fields import VectorField


def _validate_vector_fields(
    vector_fields: tuple[VectorField, ...],
) -> None:
    if not isinstance(vector_fields, tuple):
        raise TypeError("vector_fields must be a tuple.")
    if not vector_fields:
        raise ValueError("vector_fields must not be empty.")
    if any(not callable(vector_field) for vector_field in vector_fields):
        raise TypeError("every vector field must be callable.")


def _validate_state(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor.")
    if x.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {tuple(x.shape)}.")
    if x.numel() == 0:
        raise ValueError("x must contain at least one state coordinate.")
    if x.dtype != torch.float64:
        raise TypeError("x must have dtype torch.float64.")
    if not torch.isfinite(x).all():
        raise ValueError("x must contain only finite values.")


def _validate_output(output: torch.Tensor, x: torch.Tensor) -> None:
    if not isinstance(output, torch.Tensor):
        raise TypeError("a vector field must return a torch.Tensor.")
    if tuple(output.shape) != tuple(x.shape):
        raise ValueError(
            "a vector field output must have the same shape as x; "
            f"got {tuple(output.shape)} and {tuple(x.shape)}."
        )
    if output.dtype != torch.float64:
        raise TypeError("a vector field output must have dtype torch.float64.")
    if output.device != x.device:
        raise ValueError("a vector field output must be on the same device as x.")
    if not torch.isfinite(output).all():
        raise FloatingPointError(
            "a represented vector field returned a non-finite value."
        )


def _build_evaluator(
    node: LieNode,
    vector_fields: tuple[VectorField, ...],
) -> VectorField:
    if isinstance(node, Generator):
        if node.index >= len(vector_fields):
            raise ValueError(
                f"generator index {node.index} requires at least "
                f"{node.index + 1} vector fields."
            )
        base_field = vector_fields[node.index]

        def generator_evaluator(x: torch.Tensor) -> torch.Tensor:
            _validate_state(x)
            output = base_field(x)
            _validate_output(output, x)
            return output

        return generator_evaluator

    if not isinstance(node, Bracket):
        raise TypeError("node must be a Generator or Bracket.")

    left_evaluator = _build_evaluator(node.left, vector_fields)
    right_evaluator = _build_evaluator(node.right, vector_fields)
    left_jacobian = torch.func.jacrev(left_evaluator)
    right_jacobian = torch.func.jacrev(right_evaluator)

    def bracket_evaluator(x: torch.Tensor) -> torch.Tensor:
        _validate_state(x)
        left_value = left_evaluator(x)
        right_value = right_evaluator(x)
        output = (
            right_jacobian(x) @ left_value
            - left_jacobian(x) @ right_value
        )
        _validate_output(output, x)
        return output

    return bracket_evaluator


def make_lie_evaluator(
    node: LieNode,
    vector_fields: tuple[VectorField, ...],
) -> VectorField:
    """Represent ``node`` as a differentiable vector field."""
    _validate_vector_fields(vector_fields)
    if not isinstance(node, (Generator, Bracket)):
        raise TypeError("node must be a Generator or Bracket.")
    return _build_evaluator(node, vector_fields)


def evaluate_lie_node(
    node: LieNode,
    vector_fields: tuple[VectorField, ...],
    x: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one represented Lie node at one state."""
    return make_lie_evaluator(node, vector_fields)(x)


__all__ = [
    "evaluate_lie_node",
    "make_lie_evaluator",
]
