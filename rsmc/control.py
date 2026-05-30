"""Linear readout (ridge-trained) -> predictable martingale control variate.

The control is linear in the readout parameter beta, and the martingale
increments are fixed, so the whole path collapses into a single regression row:

    Phi_i = sum_{k=0}^{N-1} phi_{i,k} * dM_{i,k+1}   in R^q,

where phi_{i,k} = [1, t_k, h_{i,k}] are the (predictable) readout features and
dM are the Brownian shocks. The control for path i is then

    C_i = beta . Phi_i = sum_k (beta . phi_{i,k}) * dM_{i,k+1},

which has *population mean zero* because E[phi_k * dM_{k+1}] = 0 (phi_k is
F_{t_k}-measurable, dM_{k+1} is a mean-zero increment). Hence E[Phi] = 0 and

    E[G - C] = E[G] = p_N        (unbiased), with hopefully smaller variance.

Training (on the TRAINING sample only):

    (a_hat, beta_hat) = argmin_{a, beta}  sum_i (G_i - a - beta . Phi_i)^2 + lam ||beta||^2.

The intercept ``a`` absorbs the unknown price level p_N so it does not distort
the fitted control; it is used only for training and dropped from the control.
This equals ridge regression on centred features and centred payoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import PathFeatures
from .reservoir import RandomizedSignatureReservoir, readout_features


def control_features(readout_phi: np.ndarray, martingale_increments: np.ndarray) -> np.ndarray:
    """Collapse each path into the regression row Phi_i.

    Scalar increments dM_{k+1} (shape ``(n, N)``, e.g. Black-Scholes):

        Phi_i = sum_k phi_{i,k} * dM_{i,k+1}              in R^q.

    Vector increments dM_{k+1} in R^m (shape ``(n, N, m)``, e.g. Heston's two
    Brownian shocks): the control is C = sum_k Z_k . dM_{k+1} with one
    predictable readout direction per martingale component, Z_k^{(j)} =
    beta_j . phi_k. Collecting the blocks gives

        Phi_i = [ sum_k phi_{i,k} dM^{(1)}, ..., sum_k phi_{i,k} dM^{(m)} ]  in R^{m q},

    i.e. one q-block per martingale direction (m = 1 reduces to the scalar
    case). Only the predictable prefix phi_0..phi_{N-1} is ever used, so the
    integrand never looks ahead.

    Parameters
    ----------
    readout_phi:
        Readout features phi_k, shape ``(n, N + 1, q)``. Only the predictable
        prefix phi_0..phi_{N-1} is used.
    martingale_increments:
        Brownian shocks dM_{k+1}, shape ``(n, N)`` (scalar) or ``(n, N, m)``
        (vector).

    Returns
    -------
    np.ndarray
        Per-path regression rows Phi, shape ``(n, q)`` (scalar) or
        ``(n, m * q)`` (vector, direction-major blocks).
    """
    predictable = readout_phi[:, :-1, :]                     # phi_0 .. phi_{N-1}, (n, N, q)
    if martingale_increments.ndim == 2:                      # scalar increments (n, N)
        return np.einsum("nkq,nk->nq", predictable, martingale_increments)
    # vector increments (n, N, m): one q-block per direction, then concatenate.
    blocks = np.einsum("nkq,nkm->nmq", predictable, martingale_increments)   # (n, m, q)
    return blocks.reshape(blocks.shape[0], -1)               # (n, m * q)


def assemble_control_features(
    reservoir: RandomizedSignatureReservoir,
    features: PathFeatures,
    times: np.ndarray,
    martingale_increments: np.ndarray,
    batch_size: int = 10_000,
) -> np.ndarray:
    """Run the reservoir and build the regression rows Phi, in path-batches.

    Batching keeps memory bounded: the full reservoir-state tensor for a large
    test set would be huge, but we only ever keep one batch of states at a time
    and immediately collapse it into Phi. Scalar increments give ``q = D + 2``;
    vector increments with m directions give ``q = m * (D + 2)`` (one block each).
    """
    n_paths = features.values.shape[0]
    m = 1 if martingale_increments.ndim == 2 else martingale_increments.shape[2]
    q = (reservoir.dim + 2) * m
    phi_matrix = np.empty((n_paths, q))
    for start in range(0, n_paths, batch_size):
        stop = min(start + batch_size, n_paths)
        batch = PathFeatures(features.values[start:stop], features.names)
        states = reservoir.run(batch)
        phi = readout_features(states, times)
        phi_matrix[start:stop] = control_features(phi, martingale_increments[start:stop])
    return phi_matrix


@dataclass(frozen=True)
class RidgeMartingaleControl:
    """A ridge-trained linear readout defining the control C_i = beta . Phi_i.

    Attributes
    ----------
    beta:
        Readout coefficients in Phi-space, shape ``(q,)``.
    intercept:
        Training-only intercept a_hat (the fitted price level). NOT used in the
        control, kept only for diagnostics.
    lam:
        Ridge penalty used for the fit.
    """

    beta: np.ndarray
    intercept: float
    lam: float

    @classmethod
    def fit(cls, phi_matrix: np.ndarray, payoff: np.ndarray, lam: float) -> "RidgeMartingaleControl":
        """Ridge-fit the readout with an unpenalised intercept.

        Features are standardised internally (so ``lam`` has a consistent
        meaning across feature scales); ``beta`` is returned in the original
        Phi-space so the control can be applied directly to raw Phi.
        """
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}.")

        mean = phi_matrix.mean(axis=0)
        std = phi_matrix.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)             # guard constant columns
        standardised = (phi_matrix - mean) / std
        payoff_centred = payoff - payoff.mean()
        n_features = standardised.shape[1]

        # Ridge as augmented least squares: stack sqrt(lam) * I below the design
        # matrix and zeros below the target. Robust for lam = 0 too (SVD-based).
        design = np.vstack([standardised, np.sqrt(lam) * np.eye(n_features)])
        target = np.concatenate([payoff_centred, np.zeros(n_features)])
        theta, *_ = np.linalg.lstsq(design, target, rcond=None)

        beta = theta / std                                # back to original Phi-space
        intercept = float(payoff.mean() - beta @ mean)
        return cls(beta=beta, intercept=intercept, lam=lam)

    def control(self, phi_matrix: np.ndarray) -> np.ndarray:
        """Mean-zero control values C_i = beta . Phi_i (no intercept)."""
        return phi_matrix @ self.beta


def variance_reduction(control: RidgeMartingaleControl, phi_matrix: np.ndarray, payoff: np.ndarray) -> float:
    """Var(G) / Var(G - C) on the given sample."""
    controlled = payoff - control.control(phi_matrix)
    return float(payoff.var(ddof=1) / controlled.var(ddof=1))


def tune_ridge_lambda(
    phi_train: np.ndarray,
    payoff_train: np.ndarray,
    phi_val: np.ndarray,
    payoff_val: np.ndarray,
    lambdas: tuple[float, ...],
) -> tuple[float, list[tuple[float, float, float]]]:
    """Pick lam maximising validation variance reduction.

    Returns
    -------
    best_lambda:
        The lam with the highest validation VR.
    table:
        Rows ``(lam, train_VR, val_VR)`` for every candidate (for display).
    """
    table: list[tuple[float, float, float]] = []
    best_lambda, best_val_vr = lambdas[0], -np.inf
    for lam in lambdas:
        control = RidgeMartingaleControl.fit(phi_train, payoff_train, lam)
        train_vr = variance_reduction(control, phi_train, payoff_train)
        val_vr = variance_reduction(control, phi_val, payoff_val)
        table.append((lam, train_vr, val_vr))
        if val_vr > best_val_vr:
            best_lambda, best_val_vr = lam, val_vr
    return best_lambda, table


@dataclass(frozen=True)
class ControlReport:
    """Evaluation metrics for plain vs controlled Monte Carlo on a test sample."""

    n_eval: int
    price_mc: float
    price_cv: float
    se_mc: float
    se_cv: float
    var_mc: float
    var_cv: float
    variance_reduction: float
    se_reduction: float            # SE_MC / SE_CV = sqrt(VR)
    control_mean: float            # mean(C); should be ~ 0 (unbiasedness)
    control_mean_se: float
    price_ref: float | None = None
    price_ref_label: str | None = None

    def summary(self) -> str:
        """Human-readable multi-line report."""
        lines = [
            f"Evaluation on {self.n_eval:,} independent test paths",
            f"    plain MC   : price = {self.price_mc:.4f}   SE = {self.se_mc:.4f}",
            f"    controlled : price = {self.price_cv:.4f}   SE = {self.se_cv:.4f}",
            "",
            f"    variance reduction VR = Var(G)/Var(G-C) = {self.variance_reduction:.2f}x",
            f"    standard-error reduction SE_MC/SE_CV     = {self.se_reduction:.2f}x",
            "",
            "    bias check (controlled estimator is unbiased):",
            f"        mean(C)        = {self.control_mean:+.4f} +/- {1.96 * self.control_mean_se:.4f} (95% CI; should contain 0)",
            f"        |p_MC - p_CV|  = {abs(self.price_mc - self.price_cv):.4f}",
        ]
        if self.price_ref is not None:
            label = self.price_ref_label or "reference price"
            # Signed gaps (not |.|): when p_ref is a one-sided bound the sign is
            # the point of the check, so showing it avoids reading the gap as an error.
            lines += [
                "",
                f"    {label}: p_ref = {self.price_ref:.4f}",
                f"        p_MC - p_ref = {self.price_mc - self.price_ref:+.4f}",
                f"        p_CV - p_ref = {self.price_cv - self.price_ref:+.4f}",
            ]
        return "\n".join(lines)


def evaluate_control(
    control: RidgeMartingaleControl,
    phi_matrix: np.ndarray,
    payoff: np.ndarray,
    price_ref: float | None = None,
    price_ref_label: str | None = None,
) -> ControlReport:
    """Compare plain Monte Carlo with the controlled estimator on test paths.

    Builds Y^MC = G and Y^CV = G - C, then reports prices, standard errors,
    the variance-reduction ratio and a bias check. The headline metric is the
    standard error of the *price estimator*, not the per-path residual.
    """
    g = payoff
    c = control.control(phi_matrix)
    y_mc = g
    y_cv = g - c
    n = g.size

    var_mc = float(y_mc.var(ddof=1))
    var_cv = float(y_cv.var(ddof=1))
    se_mc = float(np.sqrt(var_mc / n))
    se_cv = float(np.sqrt(var_cv / n))

    return ControlReport(
        n_eval=n,
        price_mc=float(y_mc.mean()),
        price_cv=float(y_cv.mean()),
        se_mc=se_mc,
        se_cv=se_cv,
        var_mc=var_mc,
        var_cv=var_cv,
        variance_reduction=var_mc / var_cv,
        se_reduction=se_mc / se_cv,
        control_mean=float(c.mean()),
        control_mean_se=float(c.std(ddof=1) / np.sqrt(n)),
        price_ref=price_ref,
        price_ref_label=price_ref_label,
    )
