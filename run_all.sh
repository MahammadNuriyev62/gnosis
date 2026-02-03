#!/bin/bash
# Run all experiments sequentially, prioritized by importance.
# Priority: teachers → exp1 (core finding) → exp5 (initialization) →
#           exp2 (ensemble) → exp6 (CKA) → exp3 (augmentation) → exp4 (optimization)

set -e
cd /workspace/gnosis
source .venv/bin/activate

log() {
    echo ""
    echo "============================================"
    echo "$1"
    echo "Time: $(date)"
    echo "============================================"
}

log "Starting full experiment pipeline"

# Step 1: Train 5 teacher models (loads cached ones, trains missing)
log "STEP 1: Training/loading teacher models"
python3 run_experiments.py --exp teachers 2>&1 | tee -a full_experiment_log.txt

# Step 2: Experiment 1 - Self-distillation (3 trials) - CORE RESULT
log "STEP 2: Experiment 1 - Self-Distillation (3 trials)"
python3 run_experiments.py --exp 1 --num_trials 3 2>&1 | tee -a full_experiment_log.txt

# Step 3: Experiment 5 - Initialization (1 trial) - KEY FINDING
log "STEP 3: Experiment 5 - Initialization (1 trial)"
python3 run_experiments.py --exp 5 --num_trials 1 2>&1 | tee -a full_experiment_log.txt

# Step 4: Experiment 2 - Ensemble distillation (1 trial)
log "STEP 4: Experiment 2 - Ensemble Distillation (1 trial)"
python3 run_experiments.py --exp 2 --num_trials 1 2>&1 | tee -a full_experiment_log.txt

# Step 5: Experiment 6 - CKA analysis (1 trial)
log "STEP 5: Experiment 6 - CKA Analysis (1 trial)"
python3 run_experiments.py --exp 6 --num_trials 1 2>&1 | tee -a full_experiment_log.txt

# Step 6: Experiment 3 - Augmentation effects (1 trial)
log "STEP 6: Experiment 3 - Augmentation Effects (1 trial)"
python3 run_experiments.py --exp 3 --num_trials 1 2>&1 | tee -a full_experiment_log.txt

# Step 7: Experiment 4 - Optimization (1 trial)
log "STEP 7: Experiment 4 - Optimization (1 trial)"
python3 run_experiments.py --exp 4 --num_trials 1 2>&1 | tee -a full_experiment_log.txt

log "ALL EXPERIMENTS COMPLETE"

# Step 8: Generate plots and report values
log "Generating plots and extracting report values"
python3 plot_results.py 2>&1 | tee -a full_experiment_log.txt

log "PIPELINE COMPLETE"
