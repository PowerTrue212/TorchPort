#!/bin/bash
# RAFT training calibration batch: 11 more reference repeats (cal-r02..cal-r12),
# three negative controls, and their hidden verifies. Appends to one log.
set -uo pipefail
R=/root/autodl-tmp/RAFT-Benchmark-v1
cd "$R" || exit 1
LOG=$R/cal-r1/logs/batch-r1.log
PY=/root/autodl-tmp/envs/source/bin/python
export PYTHONPATH=$R/oracle-bundle-train-three-r1-r6/src
echo "=== batch start $(date -Is)" >> "$LOG"
df -h /root/autodl-tmp | tail -1 >> "$LOG"
for i in 02 03 04 05 06 07 08 09 10 11 12; do
  echo "=== cal-r$i start $(date -Is)" >> "$LOG"
  $PY cal-r1/scripts/run_raft_cal_verify.py "cal-r$i" controller-reference-r1 >> "$LOG" 2>&1
  echo "=== cal-r$i exit=$? $(date -Is)" >> "$LOG"
done
echo "=== negatives $(date -Is)" >> "$LOG"
$PY cal-r1/scripts/make_raft_neg_controls.py >> "$LOG" 2>&1
echo "=== negatives exit=$?" >> "$LOG"
for n in neg-gamma neg-wd neg-gru; do
  echo "=== $n start $(date -Is)" >> "$LOG"
  $PY cal-r1/scripts/run_raft_cal_verify.py "$n" "cal-r1/negatives/$n" >> "$LOG" 2>&1
  echo "=== $n exit=$? $(date -Is)" >> "$LOG"
done
echo "=== batch ALLDONE $(date -Is)" >> "$LOG"
