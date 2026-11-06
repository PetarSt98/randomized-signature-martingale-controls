"""Predict deliverable battery energy from information available at the end of the observed trace.

Inputs comprise measurements available at the quote date. No test labels are accepted. Outputs are separate frozen-draw forecasts of MWh in [0,1].
"""
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsmc.features import PathFeatures
from rsmc.reservoir import RandomizedSignatureReservoir

WIDTH = 32
GAINS = (0.3, 1.0, 2.0)
PENALTIES = (1.e-5, 0.01, 1., 100.)
METHODS = ("static", "ewma", "circuit", "reservoir")


def _scale(x, axes=0):
    return x.mean(axis=axes), np.maximum(x.std(axis=axes), 1.e-7)


def _smooth(x, half_lives):
    """Last EMA per variable, initialized by its first observation."""
    k = np.arange(x.shape[1] - 1, -1, -1)
    decay = .5 ** (1 / np.asarray(half_lives))
    w = (1 - decay)[:, None] * decay[:, None] ** k
    w[:, 0] = decay ** (x.shape[1] - 1)
    return np.einsum("ht,ntd->nhd", w, x, optimize=True).reshape(len(x), -1)


def _summaries(x, name):
    obs = x[:, :, 1:]
    if name == "static":
        return obs[:, -1]
    if name == "ewma":
        # Eight rates span the full measurement history, including the latest sample.
        return np.c_[obs[:, -1], _smooth(obs, np.geomspace(1, x.shape[1], 8))]
    # Phenomenological circuit observer. Approximate polarization and heating with
    # two relaxation rates; estimate apparent cell resistance from load transitions.
    current, voltage = x[:, :, 3], x[:, :, 2]
    di, dv = np.diff(current, axis=1), np.diff(voltage, axis=1)
    resistance = -np.sum(di * dv, axis=1) / (1.e-6 + np.sum(di ** 2, axis=1))
    proxy = np.stack((current, current ** 2), axis=-1)
    return np.c_[obs[:, -1], _smooth(proxy, (2., 12.)), resistance,
                 voltage[:, -1] + resistance * current[:, -1]]


def _project(z, seed, gain=1.0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((z.shape[1], WIDTH)) / np.sqrt(z.shape[1])
    b = rng.standard_normal(WIDTH)
    return np.tanh(gain * (z @ w) + b)


def _encode(x, seed, gain, trace_scale):
    mu, sd = trace_scale
    rs = RandomizedSignatureReservoir.draw(x.shape[2] + 1, WIDTH, np.random.default_rng(seed))
    names = tuple(f"x{k}" for k in range(x.shape[2] + 1))
    batches = []
    for start in range(0, len(x), 320):
        batch = x[start:start + 320]
        z = (batch[:, :, 1:] - mu) / sd
        ti = batch[:, :, 0]
        # Positive current denotes discharge. Current at a timestamp describes
        # the preceding interval; a trapezoidal integral only uses past readings.
        integrate = np.cumsum(.5 * (z[:, 1:, 2] + z[:, :-1, 2]) * np.diff(ti, axis=1), axis=1)
        charge = np.c_[np.zeros(len(z)), integrate]
        levels = np.concatenate((ti[:, :, None], z, charge[:, :, None]), axis=2)
        # Supply the absolute initial readings before processing their increments.
        # The origin for each sensor is its pooled training mean.
        levels = np.concatenate((np.zeros((len(z), 1, levels.shape[2])), levels), axis=1)
        batches.append(rs.run(PathFeatures(gain * levels, names))[:, -1].copy())
    return np.concatenate(batches)


def _fits(x, y):
    mu, sd = _scale(x)
    z = (x - mu) / sd
    g = z.T @ z
    rhs = z.T @ (y - y.mean())
    coeff = np.stack([solve(g + p * np.eye(len(g)), rhs, assume_a="pos") for p in PENALTIES], axis=1)
    return (coeff / sd[:, None], y.mean() - (mu / sd) @ coeff)


def fit_predict(train_x, train_y, val_x, val_y, test_x, seed):
    """Use common frozen draws for each method. Returns {method: [2, n_test]}."""
    if train_x.ndim != 3 or train_x.shape[2] not in (4, 5):
        raise ValueError("Expected [path,time,(time,SOC,voltage,C-rate[,temperature])]")
    train_y, val_y = np.ravel(train_y), np.ravel(val_y)
    splits = (train_x, val_x, test_x)
    # Same snapshot normalization, passthrough, and 32 random features for every method.
    sm, ss = _scale(train_x[:, -1, 1:])
    snap = [(x[:, -1, 1:] - sm) / ss for x in splits]
    seeds = [int(seed), int(seed) + 100003]
    common = [[np.c_[s, _project(s, k)] for s in snap] for k in seeds]
    trace_scale = _scale(train_x[:, :, 1:], axes=(0, 1))
    info, predictions = {}, {}
    cde_cache = {}

    def cde(j, k, gain):
        key = (j, k, gain)
        if key not in cde_cache:
            cde_cache[key] = _encode(splits[j], seeds[k] + 40009, gain, trace_scale)
        return cde_cache[key]

    for name in METHODS:
        before = perf_counter()
        if name != "reservoir":
            summaries = [_summaries(x, name) for x in splits]
            mu, sd = _scale(summaries[0])
            summaries = [(s - mu) / sd for s in summaries]

        def make(j, k, gain):
            key = seeds[k] + 40009
            if name == "reservoir":
                extra = cde(j, k, gain)
            else:
                extra = _project(summaries[j], key, gain)
            return np.c_[common[k][j], extra]

        losses, fitted = [], {}
        for gain in GAINS:
            draw_losses = []
            for k in range(len(seeds)):
                w, b = _fits(make(0, k, gain), train_y)
                fitted[k, gain] = w, b
                pred = np.clip(make(1, k, gain) @ w + b, 0., 1.)
                draw_losses.append(((pred - val_y[:, None]) ** 2).mean(0))
            losses.append(np.mean(draw_losses, axis=0))
        gi, pi = np.unravel_index(np.argmin(losses), (len(GAINS), len(PENALTIES)))
        gain = GAINS[gi]
        heldout = []
        for k in range(len(seeds)):
            w, b = fitted[k, gain]
            heldout.append(np.clip(make(2, k, gain) @ w[:, pi] + b[pi], 0., 1.))
        predictions[name] = np.stack(heldout)
        info[name] = dict(gain=gain, ridge=PENALTIES[pi], val_mse=float(losses[gi][pi]),
                          seconds=perf_counter()-before)
        if name == "circuit":
            circuit = [[make(j, k, gain) for j in range(3)] for k in range(len(seeds))]
    # Prespecified additional-capacity check. The circuit map at its chosen gain
    # serves as a common starting point. Each branch may add 32 frozen columns or
    # keep the original circuit regression unchanged, selected on validation.
    for name in ("circuit_plus_static", "circuit_plus_reservoir"):
        before = perf_counter()
        def joint(j, k, gain):
            extra = (cde(j, k, gain) if name == "circuit_plus_reservoir"
                     else _project(snap[j], seeds[k] + 72391, gain))
            return np.c_[circuit[k][j], extra]
        scores, coefs = [], {}
        for gain in GAINS:
            mse = []
            for k in range(len(seeds)):
                w, b = _fits(joint(0, k, gain), train_y)
                coefs[k, gain] = w, b
                pred = np.clip(joint(1, k, gain) @ w + b, 0., 1.)
                mse.append(((pred - val_y[:, None]) ** 2).mean(0))
            scores.append(np.mean(mse, axis=0))
        gi, pi = np.unravel_index(np.argmin(scores), (len(GAINS), len(PENALTIES)))
        best = float(scores[gi][pi])
        if best >= info['circuit']['val_mse']:
            predictions[name] = predictions['circuit'].copy()
            info[name] = dict(choice="base_circuit", val_mse=info['circuit']['val_mse'],
                              seconds=perf_counter()-before)
        else:
            gain = GAINS[gi]
            heldout = []
            for k in range(len(seeds)):
                w, b = coefs[k, gain]
                heldout.append(np.clip(joint(2, k, gain) @ w[:, pi] + b[pi], 0., 1.))
            predictions[name] = np.stack(heldout)
            info[name] = dict(choice="extended", gain=gain, ridge=PENALTIES[pi],
                              val_mse=best, seconds=perf_counter()-before)
    fit_predict.last_validation = info
    return predictions
