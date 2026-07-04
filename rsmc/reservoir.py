"""Frozen randomized-signature reservoir (appendix construction).

For each input channel r = 1, ..., d we draw, *once*, random parameters

    W_r in R^{D x D},   xi_r in R^D,

and freeze them (the draw theta_RS). With a bounded activation rho (default
tanh) the reservoir state evolves along each path by

    h_0   = 0 in R^D,
    h_{k+1} = h_k + sum_{r=1}^d rho(W_r h_k + xi_r) * Delta x^{(r)}_{k+1}.

Because h_k is computed only from x_0, ..., x_k, it is F_{t_k}-measurable
(non-anticipative / predictable). This is exactly the property the martingale
control variate needs, and the reason only the downstream readout is trained
while these weights stay frozen.

Readout features
----------------
The proposal's example readout feature vector is

    phi_k = [1, t_k, h_k] in R^{D + 2},

built by :func:`readout_features`. Training the linear readout beta on phi_k
is the next step of the project; here we only build the frozen feature map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .features import PathFeatures


@dataclass(frozen=True)
class RandomizedSignatureReservoir:
    """A frozen randomized-signature reservoir.

    Attributes
    ----------
    weights:
        Frozen weight matrices W_r stacked as shape ``(d, D, D)``.
    biases:
        Frozen bias vectors xi_r stacked as shape ``(d, D)``.
    activation:
        Bounded component-wise activation rho (default ``np.tanh``).
    """

    weights: np.ndarray
    biases: np.ndarray
    activation: Callable[[np.ndarray], np.ndarray] = field(default=np.tanh)

    @classmethod
    def draw(
        cls,
        n_channels: int,
        dim: int = 64,
        rng: np.random.Generator | None = None,
        weight_scale: float = 1.0,
        bias_scale: float = 1.0,
        activation: Callable[[np.ndarray], np.ndarray] = np.tanh,
    ) -> "RandomizedSignatureReservoir":
        """Draw and freeze a reservoir of dimension ``dim`` (default D = 64).

        The weights are scaled by ``1 / sqrt(D)`` so that ``W_r h`` stays O(1)
        when ``h`` is O(1); this keeps tanh in its informative (non-saturated)
        regime. ``weight_scale`` and ``bias_scale`` are the reservoir-scale
        hyper-parameters the proposal tunes on the validation set.

        Parameters
        ----------
        n_channels:
            Number of input feature channels d (must match the features later
            passed to :meth:`run`).
        dim:
            Reservoir dimension D.
        rng:
            Random generator used for the (one-off) draw. Defaults to a fresh
            ``np.random.default_rng()``.
        weight_scale, bias_scale:
            Standard-deviation multipliers for W_r and xi_r.
        activation:
            Bounded activation rho.
        """
        if rng is None:
            rng = np.random.default_rng()
        if n_channels <= 0 or dim <= 0:
            raise ValueError("n_channels and dim must be positive.")

        weights = rng.standard_normal((n_channels, dim, dim)) * (weight_scale / np.sqrt(dim))
        biases = rng.standard_normal((n_channels, dim)) * bias_scale
        return cls(weights=weights, biases=biases, activation=activation)

    @property
    def dim(self) -> int:
        """Reservoir dimension D."""
        return self.weights.shape[1]

    @property
    def n_channels(self) -> int:
        """Number of input channels d."""
        return self.weights.shape[0]

    def run(self, features: PathFeatures) -> np.ndarray:
        """Compute reservoir states h_k along every path.

        Parameters
        ----------
        features:
            Preprocessed feature paths with d channels matching the reservoir.

        Returns
        -------
        np.ndarray
            Reservoir states h_0, ..., h_N, shape ``(n_paths, N + 1, D)``.
            Note h_0 = 0 by construction.
        """
        if features.n_channels != self.n_channels:
            raise ValueError(
                f"Reservoir expects {self.n_channels} channels, "
                f"features have {features.n_channels}."
            )

        increments = features.increments              # (n, N, d)
        n_paths, n_steps, _ = increments.shape
        states = np.zeros((n_paths, n_steps + 1, self.dim))

        h = np.zeros((n_paths, self.dim))             # h_0 = 0
        for k in range(n_steps):
            step = np.zeros((n_paths, self.dim))
            for r in range(self.n_channels):
                pre_activation = h @ self.weights[r].T + self.biases[r]   # (n, D)
                channel_increment = increments[:, k, r][:, None]          # (n, 1)
                step += self.activation(pre_activation) * channel_increment
            h = h + step
            states[:, k + 1, :] = h

        return states


def readout_features(reservoir_states: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Build the readout feature vectors phi_k = [1, t_k, h_k].

    Parameters
    ----------
    reservoir_states:
        Reservoir states h_k, shape ``(n_paths, N + 1, D)``.
    times:
        Time grid (t_0, ..., t_N), shape ``(N + 1,)``.

    Returns
    -------
    np.ndarray
        Readout features phi_k, shape ``(n_paths, N + 1, D + 2)``, where the
        first column is the constant 1 and the second is the time t_k.
    """
    n_paths, n_points, _ = reservoir_states.shape
    ones = np.ones((n_paths, n_points, 1))
    time_column = np.broadcast_to(times, (n_paths, n_points))[..., None]
    return np.concatenate([ones, time_column, reservoir_states], axis=-1)
