#!/bin/bash
# Reproduces every table, figure and LaTeX number macro. Total run time about 8 minutes.
cd "$(dirname "$0")"
mkdir -p ../logs
for s in 00_build_dataset 05_changepoint 10_q1_lesson_learning 20_q2_exit_rates 21_q2_hazard 22_q2_where 30_q3_persistence_return 31_q3_concentration 40_make_figures 50_make_report_inputs; do
  echo "=== $s  $(date +%T)"
  python3 $s.py > ../logs/$s.txt 2>&1 || { echo "FAILED: $s"; tail -15 ../logs/$s.txt; exit 1; }
done
echo "ALL DONE $(date +%T)"
