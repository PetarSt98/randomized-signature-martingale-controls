"""Finite truncated-signature features - the benchmark control class.

The randomized-signature reservoir is meant to be a *cheap* stand-in for the
finite truncated signature. To benchmark it we compute the real signature and
run it through the same readout, the same ridge fit and the same evaluation -
only the path feature map changes (reservoir state ``h_k`` -> signature ``s_k``).

Signature recap
---------------
The depth-``L`` truncated signature of a path collects its iterated integrals
up to level ``L``. For a piecewise-linear path it is the product, in the
truncated tensor algebra, of the exponentials of the increments (Chen's
identity):

    S(x_0:k) = exp(dx_1) (x) ... (x) exp(dx_k),
    exp(dx)  = (1, dx, dx^{(x)2}/2!, ..., dx^{(x)L}/L!),

where ``(x)`` is the tensor (outer) product. We roll this forward to get the
running signature ``s_k = S(x_0:k)`` at every step, exactly like the reservoir
state, and use it as the predictable control feature.

Dimension (independent of the number of time steps N): for ``d`` channels,
level ``l`` contributes ``d**l`` coordinates, so

    q_L = d + d**2 + ... + d**L          (levels 1..L; level 0 is the constant 1).

We drop level 0 because the readout phi_k = [1, t_k, s_k] already has a
constant term.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .features import PathFeatures


def signature_dim(n_channels: int, level: int, include_level0: bool = False) -> int:
    """Number of truncated-signature coordinates for ``d`` channels up to ``L``.

    ``q_L = sum_{l} d**l`` over ``l = 1..L`` (or ``0..L`` if ``include_level0``).
    """
    start = 0 if include_level0 else 1
    return sum(n_channels**level_index for level_index in range(start, level + 1))


def _exp_increment(increment: np.ndarray, level: int) -> list[np.ndarray]:
    """Tensor exponential of one increment.

    Returns ``[1, dx, dx^{(x)2}/2!, ..., dx^{(x)L}/L!]`` as a list of arrays of
    shape ``(n, d**l)`` for ``l = 0..L``.
    """
    n_paths = increment.shape[0]
    exponential = [np.ones((n_paths, 1))]                 # level 0
    power = np.ones((n_paths, 1))                         # dx^{(x)0}
    factorial = 1.0
    for level_index in range(1, level + 1):
        # dx^{(x)l} = outer(dx^{(x)(l-1)}, dx), flattened.
        power = (power[:, :, None] * increment[:, None, :]).reshape(n_paths, -1)
        factorial *= level_index
        exponential.append(power / factorial)
    return exponential


def _chen_product(
    signature: list[np.ndarray], exponential: list[np.ndarray], level: int
) -> list[np.ndarray]:
    """Truncated tensor-algebra product (Chen): ``new[m] = sum_i sig[i] (x) exp[m-i]``."""
    product = []
    for total in range(level + 1):
        terms = []
        for left_level in range(total + 1):
            left = signature[left_level]                  # (n, d**left_level)
            right = exponential[total - left_level]       # (n, d**(total-left_level))
            # outer product per path, left indices first, then flatten to d**total.
            terms.append((left[:, :, None] * right[:, None, :]).reshape(left.shape[0], -1))
        product.append(sum(terms))
    return product


def _iterate_running_signature(increments: np.ndarray, level: int) -> Iterator[np.ndarray]:
    """Yield the running signature ``s_k`` (levels 1..L, concatenated) for ``k = 0..N-1``.

    ``s_k`` is the signature of the prefix ``x_0..x_k`` (using ``increments[:, :k]``);
    in particular ``s_0`` is the trivial signature (all zeros above level 0).
    """
    n_paths, n_steps, n_channels = increments.shape
    signature = [np.ones((n_paths, 1))]
    signature += [np.zeros((n_paths, n_channels**level_index)) for level_index in range(1, level + 1)]

    for step in range(n_steps):
        yield np.concatenate(signature[1:], axis=1)       # s_k, excluding the constant level 0
        exponential = _exp_increment(increments[:, step, :], level)
        signature = _chen_product(signature, exponential, level)


def running_signature(increments: np.ndarray, level: int) -> np.ndarray:
    """All running signatures ``s_0, ..., s_{N-1}``, shape ``(n, N, q_L)``.

    Materialises every state, so use it on modest path counts (tests, plots).
    For large samples use :func:`assemble_signature_control_features`, which
    accumulates the control rows on the fly.
    """
    return np.stack(list(_iterate_running_signature(increments, level)), axis=1)


def assemble_signature_control_features(
    features: PathFeatures,
    times: np.ndarray,
    martingale_increments: np.ndarray,
    level: int,
    batch_size: int = 10_000,
) -> np.ndarray:
    """Collapse each path into the signature control row ``Phi_i``.

    Mirrors :func:`rsmc.control.assemble_control_features` but uses the running
    truncated signature ``s_k`` as the feature instead of the reservoir state:

        Phi_i = sum_{k=0}^{N-1} phi_{i,k} * dM_{i,k+1},   phi_k = [1, t_k, s_k].

    Scalar increments give ``q = 2 + signature_dim(d, level)``; vector increments
    with ``m`` directions give ``q = m * (2 + signature_dim(d, level))`` with one
    block per direction (same direction-major layout as
    :func:`rsmc.control.control_features`).
    """
    increments = features.increments                      # (n, N, d)
    n_paths, _, n_channels = increments.shape
    m = 1 if martingale_increments.ndim == 2 else martingale_increments.shape[2]
    q_phi = 2 + signature_dim(n_channels, level)
    q = q_phi * m
    phi_matrix = np.empty((n_paths, q))

    for start in range(0, n_paths, batch_size):
        stop = min(start + batch_size, n_paths)
        batch = stop - start
        shocks = martingale_increments[start:stop]        # (batch, N) or (batch, N, m)
        row = np.zeros((batch, q))
        for step, signature_state in enumerate(
            _iterate_running_signature(increments[start:stop], level)
        ):
            ones = np.ones((batch, 1))
            time_column = np.full((batch, 1), times[step])
            phi = np.concatenate([ones, time_column, signature_state], axis=1)  # (batch, q_phi)
            if shocks.ndim == 2:                          # scalar increments
                row += phi * shocks[:, step][:, None]
            else:                                         # vector: one block per direction
                row += (phi[:, None, :] * shocks[:, step, :][:, :, None]).reshape(batch, -1)
        phi_matrix[start:stop] = row

    return phi_matrix
