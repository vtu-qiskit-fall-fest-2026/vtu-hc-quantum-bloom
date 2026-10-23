from pathlib import Path
import argparse
import importlib.util
import json

import numpy as np
import pandas as pd
from scipy.stats import binom
from qtrace.policies.mitigation import physical_risk


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "mission18", HERE / "18_prior_ablation.py"
)
mission18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mission18)

BATCH = mission18.BATCH
M2 = mission18.M2




def verify(plan):
    p = plan.p
    prior = plan.prior
    attenuation = 1.0 - 2.0 * p
    budget = plan.budget

    # joint[k, j] = probability of reaching count k
    # with latent noise grid index j, WITHOUT having stopped.
    joint = prior[None, :].copy()
    transition = binom.pmf(
        np.arange(BATCH + 1)[:, None], BATCH, p
    )

    total_risk = 0.0
    total_stopped = 0.0
    stopping_rows = []

    for t in range(plan.last + 1):
        table = plan.tables[t]
        gain = table["gain"]
        stop = ~table["continue"]
        n_app = budget - t * BATCH

        risk = physical_risk(
            gain[:, None], attenuation[None, :], n_app
        )
        stopped_joint = joint * stop[:, None]
        stopped_probability = float(stopped_joint.sum())
        risk_contribution = float((stopped_joint * risk).sum())

        total_stopped += stopped_probability
        total_risk += risk_contribution
        stopping_rows.append({
            "prior": plan.name,
            "budget": budget,
            "diagnostic_shots": t * BATCH,
            "stop_probability": stopped_probability,
            "risk_contribution": risk_contribution,
        })

        alive = joint * (~stop)[:, None]
        if t == plan.last:
            if alive.sum() > 1e-12:
                raise AssertionError("Unterminated probability mass.")
            break

        next_joint = np.zeros((len(joint) + BATCH, len(p)))
        for increment in range(BATCH + 1):
            next_joint[increment:increment + len(joint)] += (
                alive * transition[increment][None, :]
            )
        joint = next_joint

    # Independently integrate all deterministic stopping schedules.
    fixed_rows = []
    for t in range(plan.last + 1):
        n_diag = t * BATCH
        counts = np.arange(n_diag + 1)
        joint_fixed = (
            binom.pmf(counts[:, None], n_diag, p)
            * prior[None, :]
        )
        gain = plan.tables[t]["gain"]
        risk = physical_risk(
            gain[:, None],
            attenuation[None, :],
            budget - n_diag,
        )
        fixed_rows.append({
            "prior": plan.name,
            "budget": budget,
            "diagnostic_shots": n_diag,
            "exact_risk": float((joint_fixed * risk).sum()),
        })

    best = min(fixed_rows, key=lambda row: row["exact_risk"])
    backward_risk = float(plan.prior_risk)

    np.testing.assert_allclose(
        total_stopped, 1.0, rtol=0, atol=1e-10,
        err_msg="Stopping probabilities do not sum to one.",
    )
    np.testing.assert_allclose(
        total_risk, backward_risk, rtol=1e-9, atol=1e-11,
        err_msg="Forward physical risk disagrees with DP value.",
    )
    if total_risk > best["exact_risk"] + 1e-10:
        raise AssertionError("Adaptive is worse than best fixed under its prior.")

    summary = {
        "prior": plan.name,
        "budget": budget,
        "dp_backward_risk": backward_risk,
        "forward_exact_risk": total_risk,
        "absolute_difference": abs(total_risk - backward_risk),
        "stopped_probability": total_stopped,
        "best_fixed_shots": best["diagnostic_shots"],
        "best_fixed_risk": best["exact_risk"],
        "relative_risk_reduction_pct": (
            100.0 * (1.0 - total_risk / best["exact_risk"])
        ),
        "status": "PASS",
    }
    return summary, fixed_rows, stopping_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/dp_verification_19")
    args = parser.parse_args()
    output = Path(args.output)

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")

    summaries, fixed, stopping = [], [], []
    for prior in ["old", "effective"]:
        for budget in [128, 512]:
            plan = mission18.Plan(prior, budget)
            summary, fixed_rows, stopping_rows = verify(plan)
            summaries.append(summary)
            fixed.extend(fixed_rows)
            stopping.extend(stopping_rows)
            print(
                f"PASS: prior={prior}, budget={budget}, "
                f"forward={summary['forward_exact_risk']:.12f}, "
                f"DP={summary['dp_backward_risk']:.12f}",
                flush=True,
            )

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(fixed).to_csv(output / "all_fixed.csv", index=False)
    pd.DataFrame(stopping).to_csv(output / "stopping.csv", index=False)
    (output / "config.json").write_text(json.dumps({
        "method": "Exact forward propagation over latent noise and counts",
        "batch": BATCH,
        "target_second_moment": M2,
        "assumptions": [
            "Noise sampled from each policy's own discrete prior.",
            "Targets independent of noise and calibration.",
            "Target second moment equals the policy's frozen M2.",
            "Independent binomial calibration and application measurements.",
        ],
        "scope": [
            "Checks Plan stopping decisions and gain tables.",
            "Does not check Aer integration or hardware performance.",
            "Does not establish robustness under other noise distributions.",
        ],
    }, indent=2))

    print("\n", pd.DataFrame(summaries).to_string(index=False))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()