
#!/usr/bin/env python3
"""M32: Drift-aware quantum error decoder benchmark.

Offline experiment using the ideal stochastic X-error model.
No IBM hardware calls are made.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math

from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from qtrace.qec.drift import fisher_p_two_sided, detect_shift


ROOT = Path(__file__).resolve().parents[1]
M31 = ROOT / "results" / "policy_hardware_31"

STANDARD = np.array([0, 1, 4, 2], dtype=int)
ALTERNATIVE = STANDARD ^ 7

METHODS = [
    "standard",
    "fixed_greedy",
    "fixed_guarded95",
    "fixed_cost5",
    "periodic_cost5",
    "drift_cost5",
    "oracle",
]


# ==========================================
# 1. QUANTUM ERROR MODEL
# ==========================================

def syndrome(mask: int) -> int:
    return (
        ((mask & 1) ^ ((mask >> 1) & 1))
        + 2 * (
            ((mask >> 1) & 1)
            ^ ((mask >> 2) & 1)
        )
    )


def distribution(ratio: float) -> np.ndarray:
    """Error distribution matching M31."""
    if not 0 <= ratio <= 1:
        raise ValueError("ratio outside [0,1]")

    return np.asarray([
        .67,
        .18 * (1 - ratio),
        .04,
        .03,
        .04,
        .03,
        .18 * ratio,
        .01,
    ], dtype=float)


def schedules(blocks: int) -> dict[str, list[float]]:
    """Noise evolution scenarios."""
    if blocks % 4 or blocks < 16:
        raise ValueError(
            "blocks must be a multiple of 4 and >=16"
        )

    q = blocks // 4

    return {
        "stationary_standard": [.25] * blocks,
        "stationary_strong": [.75] * blocks,

        "step_up_down":
            [.25] * q
            + [.75] * q
            + [.25] * q
            + [.75] * q,

        "weak_interlude":
            [.25] * q
            + [.55] * q
            + [.75] * q
            + [.25] * q,

        "gradual_up":
            [.25] * q
            + list(np.linspace(.25, .75, 2 * q))
            + [.75] * q,
    }


# ==========================================
# 2. BAYESIAN ERROR DECODER
# ==========================================

@lru_cache(maxsize=None)
def confidence(a: int, b: int) -> float:
    """
    P(theta > 0.5 | Beta(a+1, b+1)).
    """
    a, b = int(a), int(b)

    if a < 0 or b < 0:
        raise ValueError("negative counts")

    n = a + b + 1

    return float(
        sum(math.comb(n, k) for k in range(a + 1))
        / (1 << n)
    )


def policy(
    counts: np.ndarray,
    method: str
) -> np.ndarray:

    counts = np.asarray(counts, dtype=int)

    if counts.shape != (8,) or np.any(counts < 0):
        raise ValueError(
            "expected eight non-negative counts"
        )

    table = STANDARD.copy()

    if method == "standard":
        return table

    for s in range(4):

        a = int(counts[ALTERNATIVE[s]])
        b = int(counts[STANDARD[s]])

        if method == "greedy":
            switch = a > b

        elif method == "guarded95":
            switch = confidence(a, b) >= .95

        elif method == "cost5":

            mu = (a + 1) / (a + b + 2)
            tail = confidence(a, b)

            gain = max(
                0.0,
                2 * mu * confidence(a + 1, b) - tail
            )

            harm = max(
                0.0,
                gain - (2 * mu - 1)
            )

            switch = gain > 5 * harm + 1e-12

        else:
            raise ValueError(
                f"unknown policy {method}"
            )

        if switch:
            table[s] = ALTERNATIVE[s]

    return table


def success(
    p: np.ndarray,
    table: np.ndarray
) -> float:
    """
    Exact average success over six logical states.
    Ideal stochastic X-error channel.
    Not hardware-measured success.
    """
    table = np.asarray(table, dtype=int)

    if (
        table.shape != (4,)
        or any(
            syndrome(int(m)) != s
            for s, m in enumerate(table)
        )
    ):
        raise ValueError("invalid syndrome decoder")

    return (
        1 + 2 * float(np.asarray(p)[table].sum())
    ) / 3


def oracle(p: np.ndarray) -> np.ndarray:
    """
    Theoretical best decoder with known true noise.
    Not implementable without privileged information.
    """
    return np.where(
        p[ALTERNATIVE] > p[STANDARD],
        ALTERNATIVE,
        STANDARD
    )


# ==========================================
# 3. NOISE DRIFT DETECTION
# ==========================================





# ==========================================
# 4. LOAD IBM M31 CALIBRATION
# ==========================================

def m31_counts(
    root: Path,
    environment: str
) -> np.ndarray:
    """
    Extract observed calibration counts.
    Does not use injected error labels.
    """

    rows = json.loads(
        (root / "cal_results.json").read_text()
    )

    counts = np.zeros(8, dtype=int)
    n = 0

    for row in rows:

        if row["environment"] != environment:
            continue

        if (
            sum(row["joint_counts"].values())
            != row["shots"]
        ):
            raise ValueError(
                "bad calibration shot count"
            )

        n += row["shots"]

        for observed, frequency in (
            row["joint_counts"].items()
        ):

            s, mask = map(
                int,
                observed.split(":")
            )

            if syndrome(mask) == s:
                counts[mask] += int(frequency)

    if n != 32:
        raise ValueError(
            f"M31 expected 32 shots, got {n}"
        )

    return counts


def verify_m31(root: Path = M31) -> dict:
    """
    Verify M31 hashes and locked decisions.
    """

    def get(name):
        return json.loads(
            (root / name).read_text()
        )

    def sha(name):
        return hashlib.sha256(
            (root / name).read_bytes()
        ).hexdigest()

    protocol, lock, decoder, job = map(
        get,
        [
            "protocol.json",
            "policy_lock.json",
            "decoder_lock.json",
            "test_job.json",
        ]
    )

    checks = {
        "source_m24_matches_protocol":
            hashlib.sha256(
                (
                    ROOT / "notebooks"
                    / "24_learned_hardware.py"
                ).read_bytes()
            ).hexdigest()
            == protocol["m24_sha256"],

        "source_m31_matches_protocol":
            hashlib.sha256(
                (
                    ROOT / "notebooks"
                    / "31_policy_hardware.py"
                ).read_bytes()
            ).hexdigest()
            == protocol["m31_sha256"],

        "calibration_hash_verified":
            sha("cal_results.json")
            == lock["cal_results_sha256"],

        "decoder_hash_verified":
            sha("decoder_lock.json")
            == lock["decoder_lock_sha256"],

        "test_plan_hash_verified":
            sha("test_plan.json")
            == job["plan_sha256"],

        "qpy_hash_verified":
            sha("test.qpy")
            == get("test_plan.json")["qpy_sha256"],
    }

    critical = [
        "calibration_hash_verified",
        "decoder_hash_verified",
        "test_plan_hash_verified",
        "qpy_hash_verified",
    ]

    if not all(checks[k] for k in critical):
        raise ValueError(
            f"M31 integrity check failed: {checks}"
        )

    summaries = {}

    for env in protocol["environments"]:

        c = m31_counts(root, env)

        actual = {
            name: policy(
                c,
                "greedy" if name == "learned" else name
            ).tolist()
            for name in [
                "standard",
                "learned",
                "guarded95",
                "cost5",
            ]
        }

        if (
            actual != lock["tables"][env]
            or actual["learned"]
            != decoder["decoders"][env]["table"]
        ):
            raise ValueError(
                "M31 locked policy disagrees "
                "with measured calibration"
            )

        summaries[env] = {
            "compatible_calibration_counts":
                c.tolist(),
            "tables": actual,
        }

    return {
        "status":
            "test_results_present"
            if (root / "test_results.json").exists()
            else "test_submitted_results_absent",

        "backend": protocol["backend"],
        "test_job_id": job["job_id"],

        "planned_total_shots":
            protocol["total_planned_shots"],

        "calibration_shots_per_environment":
            protocol[
                "calibration_shots_per_environment"
            ],

        "checks": checks,
        "environments": summaries,
    }


# ==========================================
# 5. INITIALIZE DECODER STRATEGIES
# ==========================================

def initial_tables(
    counts: np.ndarray
) -> dict[str, np.ndarray]:

    return {
        "standard": STANDARD.copy(),

        "fixed_greedy":
            policy(counts, "greedy"),

        "fixed_guarded95":
            policy(counts, "guarded95"),

        "fixed_cost5":
            policy(counts, "cost5"),

        "periodic_cost5":
            policy(counts, "cost5"),

        "drift_cost5":
            policy(counts, "cost5"),
    }


# ==========================================
# 6. RUN ONE ADAPTIVE EXPERIMENT
# ==========================================

def run_one(
    ratios: list[float],
    rng: np.random.Generator,
    init_counts: np.ndarray | None,
    monitor_shots: int,
    window: int,
    alpha: float,
    cooldown: int
):

    initial = (
        rng.multinomial(
            32,
            distribution(ratios[0])
        )
        if init_counts is None
        else init_counts.copy()
    )

    tables = initial_tables(initial)

    history = []
    samples = []
    events = []

    last_alert = -10**9
    counts_updates = Counter()

    for t, ratio in enumerate(ratios):

        p = distribution(ratio)

        # Score before observing this block.
        for method in METHODS:

            table = (
                oracle(p)
                if method == "oracle"
                else tables[method]
            )

            samples.append((
                t,
                method,
                success(p, table)
            ))

        # Monitoring observations.
        monitoring = rng.multinomial(
            monitor_shots,
            p
        )

        history.append(monitoring)

        # Periodic decoder update.
        if (t + 1) % window == 0:

            recent = np.sum(
                history[-window:],
                axis=0
            )

            tables["periodic_cost5"] = policy(
                recent,
                "cost5"
            )

            counts_updates[
                "periodic_cost5"
            ] += 1

        # Drift-aware decoder update.
        alert, pv = detect_shift(
            history,
            window,
            alpha
        )

        if (
            alert
            and t - last_alert >= cooldown
        ):

            recent = np.sum(
                history[-window:],
                axis=0
            )

            tables["drift_cost5"] = policy(
                recent,
                "cost5"
            )

            counts_updates[
                "drift_cost5"
            ] += 1

            last_alert = t

            events.append({
                "block": t,
                "p_value": pv,
                "new_decoder":
                    tables["drift_cost5"].tolist()
            })

    return (
        samples,
        events,
        dict(counts_updates)
    )


# ==========================================
# 7. EXPORT RESULTS
# ==========================================

def write_csv(
    path: Path,
    rows: list[dict],
    fields: list[str]
):

    with path.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# 8. MONTE CARLO BENCHMARK
# ==========================================

def experiment(
    *,
    seed: int,
    repeats: int,
    blocks: int,
    monitor_shots: int,
    window: int,
    alpha: float,
    cooldown: int,
    init_source: str,
    m31_root: Path,
    output: Path
) -> dict:

    if (
        repeats < 2
        or monitor_shots < 1
        or window < 2
        or cooldown < 1
        or not 0 < alpha < 1
    ):
        raise ValueError(
            "invalid experiment parameters"
        )

    if blocks < 2 * window:
        raise ValueError(
            "blocks must be >= 2*window"
        )

    audit = (
        verify_m31(m31_root)
        if init_source == "m31"
        else None
    )

    output.mkdir(
        parents=True,
        exist_ok=False
    )

    cfg = {
        "seed": seed,
        "repeats": repeats,
        "blocks": blocks,

        "monitor_shots_per_block":
            monitor_shots,

        "window_blocks":
            window,

        "test_alpha_nominal":
            alpha,

        "cooldown_blocks":
            cooldown,

        "init_source":
            init_source,

        "start_calibration_shots":
            32,

        "test_shots":
            0,

        "model":
            "ideal stochastic X; "
            "perfect gates and monitoring readout",

        "protocol_note":
            "Monitoring occurs AFTER each "
            "block's scored action; "
            "update applies next block.",

        "multiple_looks_note":
            "Nominal per-look Fisher p value; "
            "not familywise or anytime-valid.",

        "monitoring_cost_policy":
            "Periodic and drift both pay "
            "same per-block monitoring shots; "
            "fixed has none.",

        "m31_seed_note":
            "M31 uses one fixed real hardware "
            "calibration per environment "
            "in all repeats; subsequent "
            "monitoring ideal synthetic.",

        "source_sha256":
            hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
    }

    (output / "protocol.json").write_text(
        json.dumps(cfg, indent=2) + "\n"
    )

    if audit is not None:
        (
            output / "m31_audit.json"
        ).write_text(
            json.dumps(audit, indent=2) + "\n"
        )

    final = []
    trajectories = []
    events_out = []

    for scenario_i, (name, ratios) in enumerate(
        schedules(blocks).items()
    ):

        runs = {
            m: [] for m in METHODS
        }

        traces = {
            m: np.zeros(blocks)
            for m in METHODS
        }

        for repeat in range(repeats):

            rng = np.random.default_rng(
                np.random.SeedSequence([
                    seed,
                    scenario_i,
                    repeat
                ])
            )

            init = None

            if init_source == "m31":

                env = (
                    "s1_strong_alternative"
                    if ratios[0] == .75
                    else "s1_standard_favored"
                )

                init = m31_counts(
                    m31_root,
                    env
                )

            samples, events, updates = run_one(
                ratios,
                rng,
                init,
                monitor_shots,
                window,
                alpha,
                cooldown
            )

            by_method = {
                m: [] for m in METHODS
            }

            for t, m, score in samples:

                by_method[m].append(score)

                traces[m][t] += (
                    score / repeats
                )

            for m, vals in by_method.items():
                runs[m].append(
                    float(np.mean(vals))
                )

            events_out.append({
                "scenario": name,
                "repeat": repeat,
                "detected_drifts": len(events),

                "periodic_refreshes":
                    updates.get(
                        "periodic_cost5",
                        0
                    ),

                "alert_blocks": json.dumps([
                    e["block"]
                    for e in events
                ])
            })

        for method in METHODS:

            vals = np.asarray(
                runs[method]
            )

            standard = np.asarray(
                runs["standard"]
            )

            periodic = np.asarray(
                runs["periodic_cost5"]
            )

            diff_std = 100 * (
                vals - standard
            )

            diff_periodic = 100 * (
                vals - periodic
            )

            cal_shots = (
                0
                if method in [
                    "standard",
                    "oracle"
                ]
                else 32
            )

            monitor = (
                blocks * monitor_shots
                if method in [
                    "periodic_cost5",
                    "drift_cost5"
                ]
                else 0
            )

            detected_events = (
                float(np.mean([
                    e["detected_drifts"]
                    for e in events_out
                    if e["scenario"] == name
                ]))
                if method == "drift_cost5"
                else ""
            )

            final.append({
                "scenario": name,
                "method": method,
                "repeats": repeats,

                "mean_success":
                    float(vals.mean()),

                "delta_vs_standard_pp":
                    float(diff_std.mean()),

                "paired_ci95_halfwidth_pp":
                    float(
                        1.96
                        * diff_std.std(ddof=1)
                        / math.sqrt(repeats)
                    ),

                "delta_vs_periodic_pp":
                    float(diff_periodic.mean()),

                "paired_vs_periodic_ci95_halfwidth_pp":
                    float(
                        1.96
                        * diff_periodic.std(ddof=1)
                        / math.sqrt(repeats)
                    ),

                "calibration_shots":
                    cal_shots,

                "monitoring_shots":
                    monitor,

                "total_extra_shots":
                    cal_shots + monitor,

                "avg_detected_events":
                    detected_events,
            })

            for t, score in enumerate(
                traces[method]
            ):

                trajectories.append({
                    "scenario": name,
                    "block": t,
                    "ratio": ratios[t],
                    "method": method,

                    "mean_exact_success":
                        float(score),
                })

    write_csv(
        output / "summary.csv",
        final,
        list(final[0])
    )

    write_csv(
        output / "trajectory.csv",
        trajectories,
        list(trajectories[0])
    )

    write_csv(
        output / "events.csv",
        events_out,
        list(events_out[0])
    )

    return {
        "scenarios": len(
            schedules(blocks)
        ),
        "repeats": repeats,
        "summary": final,
        "output": str(output),
    }


# ==========================================
# 9. SELF TESTS
# ==========================================

def selftest() -> None:

    assert all(
        math.isclose(
            sum(distribution(r)),
            1
        )
        for r in [.25, .55, .75]
    )

    assert [
        syndrome(int(x))
        for x in STANDARD
    ] == [0, 1, 2, 3]

    assert [
        syndrome(int(x))
        for x in ALTERNATIVE
    ] == [0, 1, 2, 3]

    assert confidence(0, 0) == .5
    assert confidence(3, 0) == .9375
    assert confidence(4, 0) == .96875

    assert math.isclose(
        fisher_p_two_sided(
            0, 10, 10, 0
        ),
        2 / math.comb(20, 10)
    )

    assert (
        fisher_p_two_sided(
            5, 5, 5, 5
        )
        == 1.0
    )

    c = np.zeros(8, dtype=int)
    c[6] = 4

    assert (
        policy(c, "guarded95")[1]
        == 6
    )

    p = distribution(.75)

    assert math.isclose(
        success(p, oracle(p)),
        max(
            success(
                p,
                np.array([0, x, 4, 2])
            )
            for x in [1, 6]
        )
    )

    print("PASS: M32 pure-model self-tests")


# ==========================================
# 10. COMMAND LINE
# ==========================================

def main():

    ap = argparse.ArgumentParser(
        description=__doc__
    )

    ap.add_argument(
        "command",
        choices=[
            "selftest",
            "audit-m31",
            "run"
        ]
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=320032
    )

    ap.add_argument(
        "--repeats",
        type=int,
        default=200
    )

    ap.add_argument(
        "--smoke",
        action="store_true"
    )

    ap.add_argument(
        "--blocks",
        type=int,
        default=48
    )

    ap.add_argument(
        "--monitor-shots",
        type=int,
        default=32
    )

    ap.add_argument(
        "--window",
        type=int,
        default=4
    )

    ap.add_argument(
        "--alpha",
        type=float,
        default=.01
    )

    ap.add_argument(
        "--cooldown",
        type=int,
        default=4
    )

    ap.add_argument(
        "--init-source",
        choices=[
            "synthetic",
            "m31"
        ],
        default="synthetic"
    )

    ap.add_argument(
        "--m31-root",
        type=Path,
        default=M31
    )

    ap.add_argument(
        "--output",
        type=Path
    )

    args = ap.parse_args()

    if args.command == "selftest":
        selftest()
        return

    if args.command == "audit-m31":

        print(
            json.dumps(
                verify_m31(args.m31_root),
                indent=2
            )
        )

        return

    output = (
        args.output
        or ROOT / "results" / (
            "drift_32_smoke"
            if args.smoke
            else "drift_32"
        )
    )

    result = experiment(
        seed=args.seed,

        repeats=(
            20
            if args.smoke
            else args.repeats
        ),

        blocks=args.blocks,
        monitor_shots=args.monitor_shots,
        window=args.window,
        alpha=args.alpha,
        cooldown=args.cooldown,

        init_source=args.init_source,
        m31_root=args.m31_root,
        output=output
    )

    print(
        "Saved",
        result["output"],
        "\n"
    )

    for row in result["summary"]:

        if row["method"] in [
            "standard",
            "fixed_cost5",
            "periodic_cost5",
            "drift_cost5",
            "oracle"
        ]:

            print(
                f"{row['scenario']:20} "
                f"{row['method']:17} "
                f"success={row['mean_success']:.4f} "
                f"delta_std="
                f"{row['delta_vs_standard_pp']:+.2f} pp"
            )


if __name__ == "__main__":
    main()
