#!/bin/bash
# Recovery pipeline: runs experiments that were missed or need re-running.
# Priority: exp5 (missing lambdas) → exp6 (CKA) → exp3 → plots

set -eo pipefail
cd /workspace/gnosis
source .venv/bin/activate

log() {
    echo ""
    echo "============================================"
    echo "$1"
    echo "Time: $(date)"
    echo "============================================"
}

log "Starting recovery pipeline"

# Experiment 5 - Initialization (re-run, skips existing results)
log "Experiment 5 - Initialization (re-run missing lambdas)"
python3 run_experiments.py --exp 5 --num_trials 1 2>&1 | tee -a recovery_log.txt

# Experiment 6 - CKA analysis (if time permits)
log "Experiment 6 - CKA Analysis (1 trial)"
python3 run_experiments.py --exp 6 --num_trials 1 2>&1 | tee -a recovery_log.txt

# Experiment 3 - Augmentation (if time permits)
log "Experiment 3 - Augmentation Effects (1 trial)"
python3 run_experiments.py --exp 3 --num_trials 1 2>&1 | tee -a recovery_log.txt

log "RECOVERY COMPLETE"

# Generate plots
log "Generating plots and extracting report values"
python3 plot_results.py 2>&1 | tee -a recovery_log.txt

log "DONE"
