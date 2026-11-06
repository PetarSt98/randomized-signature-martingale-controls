"""Physics-aware benchmark: identify cell resistance and diffusion, then simulate the block.

Uses the simulator's equations, time constants, initial rest condition, parameter
bounds and dispatch contract. Uses only measured history to infer the parameters.
This benchmark answers how accurately a calibrated model can price the volume
if the true battery equations are already known.
"""
import numpy as np

from physics import CAPACITY_AH, VS, VB, _advance, _voltage, deliverable


def predict_volume(observed, step_seconds=5.):
    """Return energy prediction in MWh from raw channels in physics.CHANNELS order."""
    n = len(observed)
    # Thirty-three candidates across the simulator's prior support, fixed before evaluation.
    diffusion = 7e6 * np.linspace(.8, 2., 33) * VS * VB / (VS + VB)
    zero = np.zeros((n, len(diffusion)))
    temp = observed[:, -1:, 3]
    soc = observed[:, 0, 0:1] + zero
    state = (soc, zero, zero, zero)
    # The ohmic state for unit resistance is a known filter of observed current.
    # V(t) = Vbase(t; diffusion) - resistance * ohmic_unit(t).
    numerator, square_u, square_y = zero.copy(), zero.copy(), zero.copy()
    for j in range(1, observed.shape[1]):
        current = observed[:, j, 1:2] * CAPACITY_AH
        state = _advance(state, current, 60., (1., diffusion, temp))
        u = state[2]
        y = _voltage(state, temp) + u - observed[:, j, 2:3]
        numerator += u * y
        square_u += u * u
        square_y += y * y
    resistance = np.clip(numerator/np.maximum(square_u, 1.e-15), .085, .17)
    sse = square_y - 2.*resistance*numerator + square_u*resistance*resistance
    jj = np.argmin(sse, axis=1)
    ii = np.arange(n)
    theta_r = resistance[ii,jj]
    filtered = tuple(np.broadcast_to(s, zero.shape)[ii,jj] for s in state)
    filtered = (filtered[0], filtered[1], filtered[2]*theta_r, filtered[3])
    return deliverable(filtered, (theta_r, diffusion[jj], temp[:,0]), step_seconds)


def run():
    """Run the same three fixed test scenarios. Keep independent data generation out of inference timing."""
    from pathlib import Path
    import json
    from time import perf_counter
    from physics import simulate
    here = Path(__file__).resolve().parent
    protocol = json.loads((here / 'protocol.json').read_text())
    result_dir = here / 'results' / 'observer'
    result_dir.mkdir(exist_ok=True, parents=True)
    records = []
    for i, seeds in enumerate(protocol['physical_seeds'], 1):
        traces = simulate(protocol['test'], seeds[2])
        start = perf_counter()
        vol = np.concatenate([predict_volume(traces.observed[j:j+256])
                              for j in range(0, len(traces.volume), 256)])
        elapsed = perf_counter() - start
        # Full-state labels are used only for the test score.
        np.savez_compressed(result_dir / f'split_{i}.npz', estimated=vol, volume=traces.volume)
        record = dict(split=i, mae=float(np.mean(abs(vol-traces.volume))),
                      mse=float(np.mean((vol-traces.volume)**2)), seconds=elapsed)
        records.append(record)
        print(record,flush=True)
    (result_dir / 'results.json').write_text(json.dumps(records,indent=2))


if __name__ == '__main__':
    run()
