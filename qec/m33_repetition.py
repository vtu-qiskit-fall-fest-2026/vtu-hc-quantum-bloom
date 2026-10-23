
#!/usr/bin/env python3
"""M33: repeated-syndrome bit-flip QEC, offline benchmark + optional Qiskit Aer.

3 data qubits; syndrome measurements Z0Z1 and Z1Z2 in each round.
Phenomenological model: independent data X before each round, independent
syndrome bit readout flips, independent final data readout flips. Gates,
preparation and resets are perfect in this model. Logical states |000>,|111>.
This tests classical-basis logical memory, NOT arbitrary-state QEC.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from qtrace.qec.temporal import (
    SYNDROME,
    WEIGHT,
    validate,
    temporal_lookup,
    packed_history,
    decode_one,
)

ROOT = Path(__file__).resolve().parents[1]
STANDARD = np.array([0, 1, 4, 2], dtype=np.uint8)
METHODS = ("physical", "majority", "last_syndrome", "temporal_fixed",
           "temporal_cumulative", "temporal_adaptive", "temporal_oracle")










def simulate_batch(rng: np.random.Generator, shots: int, rounds: int,
                   p_data: float, p_syndrome: float, p_final: float):
    """Generate a common, paired test set for every policy."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    truth = rng.integers(0, 2, shots, dtype=np.uint8)
    accumulated = np.zeros(shots, dtype=np.uint8)
    history = np.zeros(shots, dtype=np.int32)
    last = np.zeros(shots, dtype=np.uint8)
    for _ in range(rounds):
        errors = rng.random((shots, 3)) < p_data
        flips = (errors @ np.array([1, 2, 4], dtype=np.uint8)).astype(np.uint8)
        accumulated ^= flips
        meas = rng.random((shots, 2)) < p_syndrome
        read_error = (meas @ np.array([1, 2], dtype=np.uint8)).astype(np.uint8)
        last = SYNDROME[accumulated] ^ read_error
        history = (history * 4 + last).astype(np.int32)
    final_noise = rng.random((shots, 3)) < p_final
    read_bits = (final_noise @ np.array([1, 2, 4], dtype=np.uint8)).astype(np.uint8)
    final = accumulated ^ (truth * 7) ^ read_bits
    return truth, history, last, final


def predictions(history: np.ndarray, last: np.ndarray, final: np.ndarray,
                rounds: int, p_fixed: float, p_estimated: float, p_cumulative: float,
                p_true: float, p_syndrome: float, p_final: float):
    return {
        "physical": final & 1,
        "majority": (WEIGHT[final] >= 2).astype(np.uint8),
        "last_syndrome": (final ^ STANDARD[last]) & 1,
        "temporal_fixed": temporal_lookup(rounds, p_fixed, p_syndrome, p_final)[
            history * 8 + final],
        "temporal_cumulative": temporal_lookup(rounds, p_cumulative, p_syndrome, p_final)[
            history * 8 + final],
        "temporal_adaptive": temporal_lookup(rounds, p_estimated, p_syndrome, p_final)[
            history * 8 + final],
        "temporal_oracle": temporal_lookup(rounds, p_true, p_syndrome, p_final)[
            history * 8 + final],
    }


def schedule(name: str, blocks: int) -> list[float]:
    if blocks < 6:
        raise ValueError("at least six blocks required")
    if name == "stationary_low":
        return [0.03] * blocks
    if name == "stationary_high":
        return [0.12] * blocks
    if name == "step_up_down":
        return [0.03 if ((i // max(1, blocks // 3)) % 2) == 0 else 0.12
                for i in range(blocks)]
    if name == "gradual_up":
        return np.linspace(0.02, 0.14, blocks).tolist()
    raise ValueError(f"unknown scenario: {name}")


def pilot_estimate(rng: np.random.Generator, p_data: float,
                   p_final: float, pilots: int) -> tuple[int, int]:
    """Independent calibration shots: known |000>, one X-noise interval,
    noisy Z measurement on 3 data qubits; NO privileged error labels.
    A pilot *shot* contributes three physical readouts; cost is pilots, not 3*pilots.
    """
    observed_flip = p_data * (1 - p_final) + (1 - p_data) * p_final
    total = 3 * pilots
    return int(rng.binomial(total, observed_flip)), total


def estimate_p(errors: int, total_bits: int, p_final: float) -> float:
    rate = (errors + 0.5) / (total_bits + 1.0)  # Jeffreys shrinkage
    return float(np.clip((rate - p_final) / (1 - 2 * p_final), .001, .49))


def snapped(p: float, grid: float = .005) -> float:
    """Quantize calibration for efficient deterministic lookup caching."""
    return round(float(np.clip(round(p / grid) * grid, .001, .49)), 3)


def experiment(*, seed: int, repeats: int, blocks: int, shots: int,
               rounds: int, p_syndrome: float, p_final: float, pilots: int,
               window: int, output: Path) -> dict:
    if repeats < 2 or blocks < 6 or shots < 1 or pilots < 1 or window < 1:
        raise ValueError("invalid benchmark run sizes")
    if not 1 <= rounds <= 7:
        raise ValueError("rounds must be 1..7")
    validate(p_syndrome, "p_syndrome")
    validate(p_final, "p_final")
    if output.exists():
        raise FileExistsError(f"output exists, refusing to overwrite: {output}")
    output.mkdir(parents=True)
    params = dict(seed=seed, repeats=repeats, blocks=blocks, shots_per_block=shots,
                  rounds=rounds, syndrome_readout_error=p_syndrome,
                  final_data_readout_error=p_final, pilot_shots_per_block=pilots,
                  adaptive_window_blocks=window, decoder_methods=METHODS,
                  note="Offline 3-data-bit quantum repetition memory, X-only noise; "
                       "perfect encoding, gates, ancilla reset; Z-basis memory test.",
                  cost="Adaptive and cumulative use identical independent pilot shots EACH block; "
                       "fixed uses pilot only at first block; oracle uses true noise "
                       "and is a bound, not deployable.",
                  causal="Pilot before scored block; no scored test label used for calibration.",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / 'protocol.json').write_text(json.dumps(params, indent=2) + '\n')

    details, summary = [], []
    scenario_names = ('stationary_low', 'stationary_high', 'step_up_down', 'gradual_up')
    for si, scen in enumerate(scenario_names):
        p_seq = schedule(scen, blocks)
        per_run = {m: [] for m in METHODS}
        for rep in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence([seed, si, rep]))
            pilot_hist = []
            errors_run = {m: 0 for m in METHODS}
            fixed_p = None
            for block, true_p in enumerate(p_seq):
                pilot_hist.append(pilot_estimate(rng, true_p, p_final, pilots))
                if fixed_p is None:
                    a, n = pilot_hist[0]
                    fixed_p = snapped(estimate_p(a, n, p_final))
                a = sum(x for x, n in pilot_hist[-window:])
                n = sum(n for x, n in pilot_hist[-window:])
                adapt_p = snapped(estimate_p(a, n, p_final))
                cumulative_a = sum(x for x, n in pilot_hist)
                cumulative_n = sum(n for x, n in pilot_hist)
                cumulative_p = snapped(estimate_p(cumulative_a, cumulative_n, p_final))
                truth, history, last, final = simulate_batch(
                    rng, shots, rounds, true_p, p_syndrome, p_final)
                guesses = predictions(history, last, final, rounds,
                                      fixed_p, adapt_p, cumulative_p, true_p, p_syndrome, p_final)
                for method, predicted in guesses.items():
                    mistakes = int(np.count_nonzero(predicted != truth))
                    errors_run[method] += mistakes
                    details.append(dict(scenario=scen, repeat=rep, block=block,
                                        method=method, true_p=true_p,
                                        estimated_p=adapt_p, cumulative_p=cumulative_p, fixed_p=fixed_p,
                                        logical_errors=mistakes, test_shots=shots))
            for method in METHODS:
                per_run[method].append(errors_run[method] / (blocks * shots))

        for method in METHODS:
            vals = np.asarray(per_run[method])
            majority = np.asarray(per_run['majority'])
            improvement = 100 * (majority - vals)  # pp reduction in logical error
            shots_pilot = (0 if method in ('physical', 'majority', 'last_syndrome',
                                           'temporal_oracle') else
                           pilots if method == 'temporal_fixed' else pilots * blocks)
            summary.append(dict(scenario=scen, method=method, repeats=repeats,
                                logical_error_rate=float(vals.mean()),
                                mean_improvement_vs_majority_pp=float(improvement.mean()),
                                paired_ci95_halfwidth_pp=float(1.96 * improvement.std(ddof=1)
                                                                / math.sqrt(repeats)),
                                pilot_shots_per_run=shots_pilot,
                                test_shots_per_run=blocks * shots))
    def csvwrite(name, rows):
        with (output / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
            writer.writerows(rows)
    csvwrite('summary.csv', summary)
    csvwrite('per_block.csv', details)
    return {'config': params, 'summary': summary, 'output': str(output)}


def aer_circuit(rounds: int, logical_bit: int):
    """Actual 5-qubit syndrome-extraction circuit. Import Qiskit only on use."""
    from qiskit import QuantumCircuit
    if not 1 <= rounds <= 7 or logical_bit not in (0, 1):
        raise ValueError('invalid circuit parameters')
    qc = QuantumCircuit(5, 2 * rounds + 3)
    if logical_bit:
        for q in range(3):
            qc.x(q)
    for t in range(rounds):
        # Aer attaches a Pauli-X error to each id operation on a data qubit.
        for q in range(3):
            qc.id(q)
        qc.cx(0, 3)
        qc.cx(1, 3)
        qc.cx(1, 4)
        qc.cx(2, 4)
        qc.measure(3, 2 * t)
        qc.measure(4, 2 * t + 1)
        qc.reset(3)
        qc.reset(4)
    for q in range(3):
        qc.measure(q, 2 * rounds + q)
    return qc


def aer_validation(*, rounds: int, shots: int, p_data: float,
                   p_syndrome: float, p_final: float, seed: int) -> dict:
    """Optional circuit-based cross-check using Aer; NEVER calls IBM hardware.

    Separate circuits prepare |000> and |111>; reconstruct syndromes from
    classical bits in Qiskit's little-endian register convention.
    """
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, pauli_error
    except ImportError as exc:
        raise RuntimeError('Install qiskit and qiskit-aer for run-aer') from exc
    for key, p in [('p_data', p_data), ('p_syndrome', p_syndrome), ('p_final', p_final)]:
        validate(p, key)
    if shots < 2:
        raise ValueError('shots >=2 required')
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(pauli_error([('X', p_data), ('I', 1-p_data)]), ['id'])
    def ro(p):
        return ReadoutError([[1-p,p], [p,1-p]])
    for qubit in (3, 4):
        noise.add_readout_error(ro(p_syndrome), [qubit])
    for qubit in (0, 1, 2):
        noise.add_readout_error(ro(p_final), [qubit])
    sim = AerSimulator(noise_model=noise, method='stabilizer')
    result = sim.run([aer_circuit(rounds, 0), aer_circuit(rounds, 1)],
                     shots=shots, seed_simulator=seed).result()
    stats = {method: 0 for method in ('physical', 'majority', 'last_syndrome', 'temporal')}
    for b in (0, 1):
        for raw, frequency in result.get_counts(b).items():
            bits = int(raw.replace(' ', ''), 2)
            hist = 0
            for t in range(rounds):
                hist = 4 * hist + ((bits >> (2*t)) & 3)
            last = (bits >> (2*(rounds-1))) & 3
            final = (bits >> (2 * rounds)) & 7
            decisions = {'physical': final & 1,
                         'majority': int(WEIGHT[final] >= 2),
                         'last_syndrome': (final ^ int(STANDARD[last])) & 1,
                         'temporal': int(temporal_lookup(rounds, p_data, p_syndrome,
                                                         p_final)[8 * hist + final])}
            for method, guess in decisions.items():
                stats[method] += int(guess != b) * frequency
    return dict(backend='AerSimulator_stabilizer', shots=2*shots, rounds=rounds,
                logical_error_rate={m: count/(2*shots) for m, count in stats.items()},
                note='Real noisy Qiskit circuits, local Aer simulation; NOT IBM hardware.')


def selftest():
    assert SYNDROME.tolist() == [0, 1, 3, 2, 2, 3, 1, 0]
    assert SYNDROME[STANDARD].tolist() == list(range(4))
    for b in (0, 1):
        for error in (0, 1, 2, 4):
            read = error ^ (7*b)
            syn = int(SYNDROME[error])
            assert (read ^ int(STANDARD[syn])) & 1 == b
            assert decode_one([syn]*3, read, .02, 0., 0.) == b
    assert decode_one([0, 0, 0], 0, .0, .0, .0) == 0
    assert decode_one([0, 0, 0], 7, .0, .0, .0) == 1
    rng = np.random.default_rng(77)
    truth, hist, last, data = simulate_batch(rng, 256, 3, 0., 0., 0.)
    decoded = predictions(hist, last, data, 3, .01, .01, .01, .01, .0, .0)
    assert all(np.array_equal(truth, x) for x in decoded.values())
    assert math.isclose(sum(temporal_lookup(2, .05, .08, .02)[0:1]),
                        int(temporal_lookup(2, .05, .08, .02)[0]))
    print('PASS M33 selftest (syndrome, X recovery, perfect channel, MAP decoder)')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['selftest', 'run', 'run-aer'])
    ap.add_argument('--seed', type=int, default=330033)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--blocks', type=int, default=12)
    ap.add_argument('--shots', type=int, default=800)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--p-syndrome', type=float, default=.08)
    ap.add_argument('--p-final', type=float, default=.04)
    ap.add_argument('--p-data', type=float, default=.06)
    ap.add_argument('--pilots', type=int, default=48)
    ap.add_argument('--window', type=int, default=2)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    if args.command == 'selftest':
        selftest()
    elif args.command == 'run-aer':
        print(json.dumps(aer_validation(rounds=args.rounds, shots=args.shots,
            p_data=args.p_data, p_syndrome=args.p_syndrome, p_final=args.p_final,
            seed=args.seed), indent=2))
    else:
        output = args.output or ROOT / 'results' / 'm33_qec_run'
        result = experiment(seed=args.seed, repeats=args.repeats, blocks=args.blocks,
            shots=args.shots, rounds=args.rounds, p_syndrome=args.p_syndrome,
            p_final=args.p_final, pilots=args.pilots, window=args.window,
            output=output)
        print('Saved results:', output)
        for row in result['summary']:
            if row['method'] in ('majority', 'temporal_fixed', 'temporal_cumulative',
                                 'temporal_adaptive', 'temporal_oracle'):
                print(f"{row['scenario']:18} {row['method']:18} "
                      f"logical_error={100*row['logical_error_rate']:.2f}% "
                      f"vs_majority={row['mean_improvement_vs_majority_pp']:+.2f} pp")


if __name__ == '__main__':
    main()
