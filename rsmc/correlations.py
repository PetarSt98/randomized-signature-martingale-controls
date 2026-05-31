"""Small correlation-matrix helpers used by the multi-factor simulators."""

from __future__ import annotations

import numpy as np


def validate_correlation_matrix(
    correlation: np.ndarray,
    size: int,
    name: str = "correlation",
) -> np.ndarray:
    """Return a validated correlation matrix as a floating NumPy array."""
    matrix = np.asarray(correlation, dtype=float)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric.")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-10):
        raise ValueError(f"{name} must have unit diagonal.")

    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite.")
    return matrix


def correlated_standard_normals(
    rng: np.random.Generator,
    n_paths: int,
    n_steps: int,
    correlation: np.ndarray,
) -> np.ndarray:
    """Draw standard normal vectors with the requested correlation matrix."""
    size = correlation.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    sqrt_correlation = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    independent = rng.standard_normal((n_paths, n_steps, size))
    return independent @ sqrt_correlation.T
