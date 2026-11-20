#!/usr/bin/env bash
# Start the two-stage arm once the GuPPy benchmark has released the CPUs.
#
# Both want the same 24 cores. Running them together would oversubscribe a 32-core
# machine and slow both, and the benchmark is nearly done, so this waits rather than
# competing. The wait is on the absence of benchmark workers, not on a fixed sleep.
set -uo pipefail
cd "$(dirname "$0")/.."

while pgrep -f "benchmark_guppy.py" > /dev/null; do sleep 60; done
echo "[queue] benchmark drained at $(date -Is); starting the two-stage arm"
exec bash scripts/run_cohort_twostage.sh
