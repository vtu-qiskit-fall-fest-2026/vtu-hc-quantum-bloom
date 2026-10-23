#!/usr/bin/env bash
set -euo pipefail

PY="${1:-python}"
OUT="$(mktemp -d /tmp/qtrace-final-XXXXXX)"

if [ "$(git branch --show-current)" != "refactor/qtrace-core" ]; then
    echo "ERROR: Expected refactor/qtrace-core branch."
    exit 1
fi

echo "Audit directory: $OUT"

echo "[1/12] Package and test suite"
"$PY" -m pip check > "$OUT/pip_check.txt"
"$PY" -m compileall -q src notebooks qec tests
"$PY" -m unittest discover \
    -s tests -p "test_*.py" -v \
    > "$OUT/tests.log" 2>&1

echo "[2/12] M20 locked benchmark: analytic smoke"
"$PY" notebooks/20_locked_benchmark.py \
    --backend analytic --smoke \
    --output "$OUT/m20" \
    > "$OUT/m20.log"

echo "[3/12] M21 repetition QEC: Aer"
"$PY" notebooks/21_repetition_qec.py \
    --mode aer --shots 128 \
    --output "$OUT/m21" \
    > "$OUT/m21.log"

echo "[4/12] M24 learned decoder: selftest"
"$PY" notebooks/24_learned_hardware.py selftest \
    > "$OUT/m24.log"

echo "[5/12] M27 generalization"
"$PY" notebooks/27_decoder_generalization.py \
    --smoke --output "$OUT/m27" \
    > "$OUT/m27.log"

echo "[6/12] M28 guarded decoder"
"$PY" notebooks/28_guarded_decoder.py \
    --smoke --output "$OUT/m28" \
    > "$OUT/m28.log"

echo "[7/12] M29 cost-sensitive decoder"
"$PY" notebooks/29_cost_sensitive_decoder.py \
    --smoke --output "$OUT/m29" \
    > "$OUT/m29.log"

echo "[8/12] M31 hardware policy selftest only"
"$PY" notebooks/31_policy_hardware.py selftest \
    > "$OUT/m31.log"

echo "[9/12] M32 selftest and synthetic drift benchmark"
"$PY" notebooks/32_drift_aware_decoder.py selftest \
    > "$OUT/m32_selftest.log"

"$PY" notebooks/32_drift_aware_decoder.py run \
    --smoke \
    --blocks 16 --monitor-shots 8 \
    --window 2 --cooldown 2 \
    --init-source synthetic \
    --output "$OUT/m32" \
    > "$OUT/m32.log"

echo "[10/12] M33 temporal MAP selftest"
"$PY" qec/m33_repetition.py selftest \
    > "$OUT/m33_selftest.log"

echo "[11/12] M33 temporal simulation"
"$PY" qec/m33_repetition.py run \
    --seed 330033 \
    --repeats 2 --blocks 6 --shots 96 \
    --rounds 3 --pilots 24 --window 2 \
    --output "$OUT/m33" \
    > "$OUT/m33.log"

echo "[12/12] M33 noisy Aer circuits"
"$PY" qec/m33_repetition.py run-aer \
    --seed 330033 --rounds 3 --shots 128 \
    > "$OUT/m33_aer.json"

echo
tail -n 4 "$OUT/tests.log"
echo
echo "PASS: All final audit commands completed."
echo "Audit logs: $OUT"

git status --short > "$OUT/git_status.txt"
git diff --name-status -- results \
    > "$OUT/results_changes.txt"

if [ -s "$OUT/results_changes.txt" ]; then
    echo "WARNING: Tracked results/ changes detected."
    cat "$OUT/results_changes.txt"
fi
