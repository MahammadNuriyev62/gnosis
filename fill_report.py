#!/usr/bin/env python3
"""
Auto-fill the LaTeX report with actual experiment results.
Reads JSON results from results/ and updates report.tex placeholders.
"""

import os
import json
import glob
import re
import subprocess
import numpy as np


RESULTS_DIR = 'results'
REPORT_PATH = 'report/report.tex'


def load_results(pattern):
    """Load all result files matching a glob pattern."""
    files = sorted(glob.glob(f'{RESULTS_DIR}/{pattern}'))
    results = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        with open(f) as fp:
            results[name] = json.load(fp)
    return results


def get_final_metrics(results, keys=None):
    """Extract final epoch metrics from multiple trial results."""
    if keys is None:
        keys = ['test_acc', 'test_ts_agree', 'test_ts_kl', 'test_ece', 'test_nll']
    metrics = {k: [] for k in keys}
    for name, records in results.items():
        if records:
            final = records[-1]
            for k in keys:
                if k in final:
                    metrics[k].append(final[k])
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items() if v}


def fmt(mean, std=None, decimals=2):
    """Format mean ± std, or just mean if std is None or zero."""
    if std is not None and std > 0:
        return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"
    return f"{mean:.{decimals}f}"


def fmt_single(value, decimals=2):
    """Format a single value."""
    return f"{value:.{decimals}f}"


def fill_report():
    """Read report.tex and replace placeholders with actual values."""
    with open(REPORT_PATH) as f:
        tex = f.read()

    replacements = {}

    # --- Experiment 2: Ensemble distillation ---
    for nt in [3, 5]:
        results = load_results(f'exp2_ensemble{nt}_trial*.json')
        if results:
            m = get_final_metrics(results)
            # Get teacher accuracy from first result
            first_rec = list(results.values())[0]
            teacher_acc = first_rec[-1].get('teacher_test_acc', 0)

            replacements[f'\\textbf{{T{nt}\\_ACC}}'] = fmt_single(teacher_acc)
            if 'test_acc' in m:
                replacements[f'\\textbf{{S{nt}\\_ACC}}'] = fmt(m['test_acc'][0], m['test_acc'][1])
            if 'test_ts_agree' in m:
                replacements[f'\\textbf{{A{nt}}}'] = fmt(m['test_ts_agree'][0], m['test_ts_agree'][1])
            if 'test_ts_kl' in m:
                replacements[f'\\textbf{{KL{nt}}}'] = fmt(m['test_ts_kl'][0], m['test_ts_kl'][1], 3)

    # --- Experiment 6: CKA ---
    rand_results = load_results('exp6_cka_rand_trial*.json')
    tinit_results = load_results('exp6_cka_tinit_trial*.json')

    if rand_results:
        m = get_final_metrics(rand_results,
                              keys=['test_ts_agree', 'test_ts_kl', 'test_cka_0', 'test_cka_1', 'test_cka_2'])
        if 'test_ts_agree' in m:
            replacements['\\textbf{R\\_AGREE}'] = fmt(m['test_ts_agree'][0], m['test_ts_agree'][1])
        if 'test_ts_kl' in m:
            replacements['\\textbf{R\\_KL}'] = fmt(m['test_ts_kl'][0], m['test_ts_kl'][1], 3)
        for i, stage in enumerate(['R\\_CKA1', 'R\\_CKA2', 'R\\_CKA3']):
            key = f'test_cka_{i}'
            if key in m:
                replacements[f'\\textbf{{{stage}}}'] = fmt(m[key][0], m[key][1], 3)

    if tinit_results:
        m = get_final_metrics(tinit_results,
                              keys=['test_ts_agree', 'test_ts_kl', 'test_cka_0', 'test_cka_1', 'test_cka_2'])
        if 'test_ts_agree' in m:
            replacements['\\textbf{T\\_AGREE}'] = fmt(m['test_ts_agree'][0], m['test_ts_agree'][1])
        if 'test_ts_kl' in m:
            replacements['\\textbf{T\\_KL}'] = fmt(m['test_ts_kl'][0], m['test_ts_kl'][1], 3)
        for i, stage in enumerate(['T\\_CKA1', 'T\\_CKA2', 'T\\_CKA3']):
            key = f'test_cka_{i}'
            if key in m:
                replacements[f'\\textbf{{{stage}}}'] = fmt(m[key][0], m[key][1], 3)

    # --- Experiment 3: Augmentation ---
    # tau=1 baseline
    t1_results = load_results('exp3_baseline_t1_trial*.json')
    if t1_results:
        m = get_final_metrics(t1_results)
        if 'test_acc' in m:
            replacements['\\textbf{AUG\\_T1\\_ACC}'] = fmt(m['test_acc'][0], m['test_acc'][1])
        if 'test_ts_agree' in m:
            replacements['\\textbf{AUG\\_T1\\_AGR}'] = fmt(m['test_ts_agree'][0], m['test_ts_agree'][1])
        if 'test_ts_kl' in m:
            replacements['\\textbf{AUG\\_T1\\_KL}'] = fmt(m['test_ts_kl'][0], m['test_ts_kl'][1], 3)

    # tau=4 baseline (same 5-teacher ensemble)
    t4_results = load_results('exp3_baseline_t4_trial*.json')
    if t4_results:
        m = get_final_metrics(t4_results)
        if 'test_acc' in m:
            replacements['\\textbf{AUG\\_T4\\_ACC}'] = fmt(m['test_acc'][0], m['test_acc'][1])
        if 'test_ts_agree' in m:
            replacements['\\textbf{AUG\\_T4\\_AGR}'] = fmt(m['test_ts_agree'][0], m['test_ts_agree'][1])
        if 'test_ts_kl' in m:
            replacements['\\textbf{AUG\\_T4\\_KL}'] = fmt(m['test_ts_kl'][0], m['test_ts_kl'][1], 3)

    # MixUp
    mu_results = load_results('exp3_mixup_t4_trial*.json')
    if mu_results:
        m = get_final_metrics(mu_results)
        if 'test_acc' in m:
            replacements['\\textbf{AUG\\_MU\\_ACC}'] = fmt(m['test_acc'][0], m['test_acc'][1])
        if 'test_ts_agree' in m:
            replacements['\\textbf{AUG\\_MU\\_AGR}'] = fmt(m['test_ts_agree'][0], m['test_ts_agree'][1])
        if 'test_ts_kl' in m:
            replacements['\\textbf{AUG\\_MU\\_KL}'] = fmt(m['test_ts_kl'][0], m['test_ts_kl'][1], 3)

    # Apply replacements
    count = 0
    for old, new in replacements.items():
        if old in tex:
            tex = tex.replace(old, new)
            count += 1
            print(f"  Replaced: {old} -> {new}")
        else:
            print(f"  NOT FOUND in tex: {old}")

    # Count remaining placeholders
    remaining = len(re.findall(r'\\textbf\{[A-Z0-9_]+\}', tex))

    with open(REPORT_PATH, 'w') as f:
        f.write(tex)

    print(f"\nApplied {count} replacements. {remaining} placeholders remaining.")
    return count, remaining


def regenerate_plots():
    """Run plot_results.py to regenerate all figures."""
    print("\nRegenerating plots...")
    result = subprocess.run(['python3', 'plot_results.py'],
                            capture_output=True, text=True, cwd='/workspace/gnosis')
    print(result.stdout)
    if result.returncode != 0:
        print("PLOT ERROR:", result.stderr)


def compile_report():
    """Compile the LaTeX report."""
    print("\nCompiling report...")
    cwd = '/workspace/gnosis/report'
    subprocess.run(['pdflatex', '-interaction=nonstopmode', 'report.tex'],
                   capture_output=True, cwd=cwd)
    subprocess.run(['bibtex', 'report'], capture_output=True, cwd=cwd)
    subprocess.run(['pdflatex', '-interaction=nonstopmode', 'report.tex'],
                   capture_output=True, cwd=cwd)
    result = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'report.tex'],
                            capture_output=True, text=True, cwd=cwd)

    for line in result.stdout.split('\n'):
        if 'Output written' in line:
            print(f"  {line.strip()}")


if __name__ == '__main__':
    print("=" * 60)
    print("AUTO-FILLING REPORT WITH EXPERIMENT RESULTS")
    print("=" * 60)

    print("\n--- Available results ---")
    for f in sorted(glob.glob(f'{RESULTS_DIR}/*.json')):
        print(f"  {os.path.basename(f)}")

    print("\n--- Filling placeholders ---")
    count, remaining = fill_report()

    regenerate_plots()
    compile_report()

    print(f"\nDone! Report compiled. {remaining} placeholders still need data.")
