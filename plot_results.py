#!/usr/bin/env python3
"""
Visualization script for "Does Knowledge Distillation Really Work?" replication.
Generates all figures for the ICML report.
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'


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


def plot_figure1():
    """
    Figure 1: Self-distillation vs Ensemble distillation.
    Agreement vs Accuracy scatter.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Self-distillation
    self_results = load_results('exp1_self_distill_trial*.json')
    if self_results:
        accs, agrees = [], []
        teacher_accs = []
        for name, records in self_results.items():
            if records:
                final = records[-1]
                accs.append(final.get('test_acc', 0))
                agrees.append(final.get('test_ts_agree', 0))
                teacher_accs.append(final.get('teacher_test_acc', 0))

        ax = axes[0]
        ax.scatter(agrees, accs, c='blue', s=80, label='Student', zorder=5)
        if teacher_accs:
            mean_teacher_acc = np.mean(teacher_accs)
            ax.axhline(y=mean_teacher_acc, color='red', linestyle='--',
                       label=f'Teacher Acc ({mean_teacher_acc:.1f}%)')
        ax.set_xlabel('Test Agreement (%)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('(a) Self-distillation')
        ax.legend()

    # (b) Ensemble distillation
    for num_t in [3, 5]:
        ens_results = load_results(f'exp2_ensemble{num_t}_trial*.json')
        if ens_results:
            accs, agrees = [], []
            teacher_accs = []
            for name, records in ens_results.items():
                if records:
                    final = records[-1]
                    accs.append(final.get('test_acc', 0))
                    agrees.append(final.get('test_ts_agree', 0))
                    teacher_accs.append(final.get('teacher_test_acc', 0))

            ax = axes[1]
            ax.scatter(agrees, accs, s=80,
                       label=f'{num_t}-teacher ensemble', zorder=5)
            if teacher_accs:
                mean_teacher_acc = np.mean(teacher_accs)
                ax.axhline(y=mean_teacher_acc, color='red', linestyle='--', alpha=0.5,
                           label=f'Teacher ({num_t}) Acc ({mean_teacher_acc:.1f}%)')

    axes[1].set_xlabel('Test Agreement (%)')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('(b) Ensemble distillation')
    axes[1].legend()

    plt.suptitle('Evaluating the Fidelity of Knowledge Distillation', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/figure1_fidelity.pdf')
    plt.savefig(f'{PLOTS_DIR}/figure1_fidelity.png')
    plt.close()
    print("Saved Figure 1")


def plot_figure3():
    """
    Figure 3: Data augmentation and distillation.
    Bar plots of accuracy and agreement for different augmentation policies.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    configs = [
        ('baseline_t1', 'Baseline(τ=1)'),
        ('baseline_t4', 'Baseline(τ=4)'),
        ('mixup_t4', 'MixUp(τ=4)'),
    ]

    accs_mean, accs_std = [], []
    agrees_mean, agrees_std = [], []
    labels = []

    for cfg_name, display_name in configs:
        results = load_results(f'exp3_{cfg_name}_trial*.json')
        if results:
            metrics = get_final_metrics(results)
            if 'test_acc' in metrics:
                accs_mean.append(metrics['test_acc'][0])
                accs_std.append(metrics['test_acc'][1])
            if 'test_ts_agree' in metrics:
                agrees_mean.append(metrics['test_ts_agree'][0])
                agrees_std.append(metrics['test_ts_agree'][1])
            labels.append(display_name)

    if not labels:
        print("No augmentation results found, skipping Figure 3")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(labels))
    width = 0.5

    # Accuracy
    colors = sns.color_palette('Set2', len(labels))
    best_idx = np.argmax(accs_mean)
    bar_colors = ['green' if i == best_idx else colors[i] for i in range(len(labels))]
    axes[0].bar(x, accs_mean, width, yerr=accs_std, color=bar_colors, capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha='right')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Test Accuracy')

    # Agreement
    best_idx = np.argmax(agrees_mean)
    bar_colors = ['green' if i == best_idx else colors[i] for i in range(len(labels))]
    axes[1].bar(x, agrees_mean, width, yerr=agrees_std, color=bar_colors, capsize=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha='right')
    axes[1].set_ylabel('Agreement (%)')
    axes[1].set_title('Teacher-Student Agreement')

    plt.suptitle('Data Augmentation and Distillation (5-teacher ensemble, CIFAR-100)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/figure3_augmentation.pdf')
    plt.savefig(f'{PLOTS_DIR}/figure3_augmentation.png')
    plt.close()
    print("Saved Figure 3")


def plot_figure6a():
    """
    Figure 6(a): Optimization experiments - SGD vs Adam at different epoch counts.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    configs = [
        ('sgd_300ep', 'SGD 300ep'),
        ('sgd_600ep', 'SGD 600ep'),
        ('adam_300ep', 'Adam 300ep'),
        ('adam_600ep', 'Adam 600ep'),
    ]

    agrees_mean, agrees_std = [], []
    labels = []

    for cfg_name, display_name in configs:
        results = load_results(f'exp4_{cfg_name}_trial*.json')
        if results:
            metrics = get_final_metrics(results)
            if 'test_ts_agree' in metrics:
                agrees_mean.append(metrics['test_ts_agree'][0])
                agrees_std.append(metrics['test_ts_agree'][1])
                labels.append(display_name)

    if not labels:
        print("No optimization results found, skipping Figure 6a")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(labels))
    colors = ['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e']
    hatches = ['', '///', '', '///']
    bars = ax.bar(x, agrees_mean, 0.5, yerr=agrees_std, color=colors[:len(labels)],
                  capsize=5, edgecolor='black')
    for bar, h in zip(bars, hatches[:len(labels)]):
        bar.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Train Agreement (%)')
    ax.set_title('Optimizer Effect on Train Agreement\n(Self-distillation, PreResNet-20, CIFAR-100)')

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/figure6a_optimization.pdf')
    plt.savefig(f'{PLOTS_DIR}/figure6a_optimization.png')
    plt.close()
    print("Saved Figure 6a")


def plot_figure6b():
    """
    Figure 6(b): Initialization experiments - lambda sweep.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    lambdas = [0.0, 0.25, 0.375, 0.5, 1.0]
    train_agrees_mean, train_agrees_std = [], []
    train_loss_mean, train_loss_std = [], []
    valid_lambdas = []

    for lam in lambdas:
        results = load_results(f'exp5_init_lam{lam}_trial*.json')
        if results:
            metrics_agree = get_final_metrics(results, keys=['train_ts_agree'])
            metrics_loss = get_final_metrics(results, keys=['train_loss'])
            if 'train_ts_agree' in metrics_agree:
                train_agrees_mean.append(metrics_agree['train_ts_agree'][0])
                train_agrees_std.append(metrics_agree['train_ts_agree'][1])
                valid_lambdas.append(lam)
            if 'train_loss' in metrics_loss:
                train_loss_mean.append(metrics_loss['train_loss'][0])
                train_loss_std.append(metrics_loss['train_loss'][1])

    if not valid_lambdas:
        print("No initialization results found, skipping Figure 6b")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Train Loss
    if train_loss_mean:
        axes[0].errorbar(valid_lambdas[:len(train_loss_mean)], train_loss_mean,
                         yerr=train_loss_std, marker='o', capsize=5)
        axes[0].set_xlabel('λ (interpolation parameter)')
        axes[0].set_ylabel('Train Loss')
        axes[0].set_title('Final Train Loss')

    # Train Agreement
    axes[1].errorbar(valid_lambdas, train_agrees_mean, yerr=train_agrees_std,
                     marker='o', capsize=5)
    axes[1].set_xlabel('λ (interpolation parameter)')
    axes[1].set_ylabel('Train Agreement (%)')
    axes[1].set_title('Final Train Agreement')

    plt.suptitle('Initialization Effect on Distillation\n'
                 '(θ_s = λ·θ_t + (1-λ)·θ_r, Self-distillation, PreResNet-20)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/figure6b_initialization.pdf')
    plt.savefig(f'{PLOTS_DIR}/figure6b_initialization.png')
    plt.close()
    print("Saved Figure 6b")


def plot_table1():
    """
    Table 1: CKA analysis - random vs teacher initialization.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    rand_results = load_results('exp6_cka_rand_trial*.json')
    tinit_results = load_results('exp6_cka_tinit_trial*.json')

    if not rand_results or not tinit_results:
        print("No CKA results found, skipping Table 1")
        return

    rand_metrics = get_final_metrics(rand_results,
                                     keys=['test_acc', 'test_ts_agree', 'test_ts_kl',
                                           'test_cka_0', 'test_cka_1', 'test_cka_2'])
    tinit_metrics = get_final_metrics(tinit_results,
                                      keys=['test_acc', 'test_ts_agree', 'test_ts_kl',
                                            'test_cka_0', 'test_cka_1', 'test_cka_2'])

    print("\n" + "=" * 70)
    print("TABLE 1: CKA Analysis (Random Init vs Teacher Init)")
    print("=" * 70)
    print(f"{'Metric':<20} {'Random Init':<25} {'Teacher Init':<25}")
    print("-" * 70)
    for key in ['test_acc', 'test_ts_agree', 'test_ts_kl',
                'test_cka_0', 'test_cka_1', 'test_cka_2']:
        if key in rand_metrics and key in tinit_metrics:
            rm, rs = rand_metrics[key]
            tm, ts = tinit_metrics[key]
            print(f"{key:<20} {rm:.3f} ± {rs:.3f}          {tm:.3f} ± {ts:.3f}")

    # Save as figure/table
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    headers = ['Init.', 'Agree. (↑)', 'KL (↓)', 'CKA Stage 1', 'CKA Stage 2', 'CKA Stage 3']
    row_labels = ['Random', 'Teacher']
    cell_data = []
    for metrics in [rand_metrics, tinit_metrics]:
        row = []
        for key in ['test_ts_agree', 'test_ts_kl', 'test_cka_0', 'test_cka_1', 'test_cka_2']:
            if key in metrics:
                m, s = metrics[key]
                row.append(f'{m:.3f} ({s:.3f})')
            else:
                row.append('N/A')
        cell_data.append(row)

    table = ax.table(cellText=cell_data, rowLabels=row_labels,
                     colLabels=headers[1:], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Table 1: Random vs Teacher Initialization\n'
                 '(PreResNet-20 self-distillation on CIFAR-100)', fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/table1_cka.pdf')
    plt.savefig(f'{PLOTS_DIR}/table1_cka.png')
    plt.close()
    print("Saved Table 1")


def plot_training_curves():
    """Plot training curves showing convergence of accuracy and agreement."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Self-distillation training curves
    results = load_results('exp1_self_distill_trial*.json')
    if not results:
        print("No self-distill results for training curves, skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for name, records in results.items():
        epochs = [r.get('epoch', i) for i, r in enumerate(records)]
        accs = [r.get('test_acc', 0) for r in records]
        agrees = [r.get('test_ts_agree', 0) for r in records]
        kls = [r.get('test_ts_kl', 0) for r in records]

        axes[0].plot(epochs, accs, alpha=0.7, label=name.split('_')[-1])
        axes[1].plot(epochs, agrees, alpha=0.7)
        axes[2].plot(epochs, kls, alpha=0.7)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Student Accuracy')
    axes[0].legend()

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Agreement (%)')
    axes[1].set_title('Teacher-Student Agreement')

    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL(T||S)')
    axes[2].set_title('Predictive KL Divergence')

    plt.suptitle('Self-Distillation Training Curves (PreResNet-20, CIFAR-100)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/training_curves.pdf')
    plt.savefig(f'{PLOTS_DIR}/training_curves.png')
    plt.close()
    print("Saved training curves")


def plot_summary_table():
    """Create a comprehensive summary table of all experiments."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    experiments = [
        ('Self-Distillation (1T)', 'exp1_self_distill_trial*.json'),
        ('Ensemble 3T', 'exp2_ensemble3_trial*.json'),
        ('Ensemble 5T', 'exp2_ensemble5_trial*.json'),
        ('Baseline τ=1', 'exp3_baseline_t1_trial*.json'),
        ('Baseline τ=4', 'exp3_baseline_t4_trial*.json'),
        ('MixUp τ=4', 'exp3_mixup_t4_trial*.json'),
    ]

    rows = []
    for exp_name, pattern in experiments:
        results = load_results(pattern)
        if results:
            metrics = get_final_metrics(results)
            row = [exp_name]
            for key in ['test_acc', 'test_ts_agree', 'test_ts_kl', 'test_ece']:
                if key in metrics:
                    m, s = metrics[key]
                    row.append(f'{m:.2f} ± {s:.2f}')
                else:
                    row.append('N/A')
            rows.append(row)

    if not rows:
        print("No results for summary table, skipping")
        return

    fig, ax = plt.subplots(figsize=(12, max(3, len(rows) * 0.6 + 1.5)))
    ax.axis('off')
    headers = ['Experiment', 'Accuracy (%)', 'Agreement (%)', 'KL(T||S)', 'ECE']
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    # Header styling
    for j in range(len(headers)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('Summary of Experimental Results', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/summary_table.pdf')
    plt.savefig(f'{PLOTS_DIR}/summary_table.png')
    plt.close()
    print("Saved summary table")


def plot_temperature_sweep():
    """Plot temperature sweep results."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    temps = [1.0, 2.0, 4.0, 8.0, 16.0]
    accs_mean, accs_std = [], []
    agrees_mean, agrees_std = [], []
    valid_temps = []

    for temp in temps:
        results = load_results(f'exp_temp{temp}_trial*.json')
        if results:
            metrics = get_final_metrics(results)
            if 'test_acc' in metrics and 'test_ts_agree' in metrics:
                accs_mean.append(metrics['test_acc'][0])
                accs_std.append(metrics['test_acc'][1])
                agrees_mean.append(metrics['test_ts_agree'][0])
                agrees_std.append(metrics['test_ts_agree'][1])
                valid_temps.append(temp)

    if not valid_temps:
        print("No temperature sweep results found, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].errorbar(valid_temps, accs_mean, yerr=accs_std, marker='o', capsize=5)
    axes[0].set_xscale('log', base=2)
    axes[0].set_xlabel('Temperature (τ)')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Student Accuracy vs Temperature')
    axes[0].xaxis.set_major_formatter(ticker.ScalarFormatter())

    axes[1].errorbar(valid_temps, agrees_mean, yerr=agrees_std, marker='o', capsize=5, color='orange')
    axes[1].set_xscale('log', base=2)
    axes[1].set_xlabel('Temperature (τ)')
    axes[1].set_ylabel('Agreement (%)')
    axes[1].set_title('Teacher-Student Agreement vs Temperature')
    axes[1].xaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.suptitle('Effect of Temperature on Self-Distillation (PreResNet-20, CIFAR-100)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/temperature_sweep.pdf')
    plt.savefig(f'{PLOTS_DIR}/temperature_sweep.png')
    plt.close()
    print("Saved temperature sweep")


def print_report_values():
    """Print all numeric values needed for the LaTeX report."""
    print("\n" + "=" * 70)
    print("VALUES FOR LATEX REPORT")
    print("=" * 70)

    # Teacher accuracy
    teacher_results = load_results('teacher_0_records.json')
    if teacher_results:
        for name, recs in teacher_results.items():
            if recs:
                final_acc = recs[-1].get('test_acc', 0)
                print(f"\nTeacher accuracy: {final_acc:.2f}%")

    # Experiment 1: Self-distillation
    exp1 = load_results('exp1_self_distill_trial*.json')
    if exp1:
        m = get_final_metrics(exp1)
        print(f"\n--- Experiment 1: Self-distillation ---")
        for k, (mean, std) in m.items():
            print(f"  {k}: {mean:.2f} ± {std:.2f}")

    # Experiment 2: Ensemble distillation
    for nt in [1, 3, 5]:
        exp2 = load_results(f'exp2_ensemble{nt}_trial*.json')
        if exp2:
            m = get_final_metrics(exp2)
            print(f"\n--- Experiment 2: {nt}-teacher ensemble ---")
            for k, (mean, std) in m.items():
                print(f"  {k}: {mean:.2f} ± {std:.2f}")

    # Experiment 3: Augmentation
    for cfg in ['baseline_t1', 'baseline_t4', 'mixup_t4']:
        exp3 = load_results(f'exp3_{cfg}_trial*.json')
        if exp3:
            m = get_final_metrics(exp3)
            print(f"\n--- Experiment 3: {cfg} ---")
            for k, (mean, std) in m.items():
                print(f"  {k}: {mean:.2f} ± {std:.2f}")

    # Experiment 4: Optimization
    for cfg in ['sgd_300ep', 'sgd_600ep', 'adam_300ep', 'adam_600ep']:
        exp4 = load_results(f'exp4_{cfg}_trial*.json')
        if exp4:
            m = get_final_metrics(exp4, keys=['train_ts_agree', 'test_ts_agree', 'test_acc'])
            print(f"\n--- Experiment 4: {cfg} ---")
            for k, (mean, std) in m.items():
                print(f"  {k}: {mean:.2f} ± {std:.2f}")

    # Experiment 5: Initialization
    print(f"\n--- Experiment 5: Initialization ---")
    for lam in [0.0, 0.25, 0.375, 0.5, 1.0]:
        exp5 = load_results(f'exp5_init_lam{lam}_trial*.json')
        if exp5:
            m = get_final_metrics(exp5, keys=['train_ts_agree', 'train_loss', 'test_ts_agree', 'test_acc'])
            vals = {k: f"{mean:.2f}" for k, (mean, std) in m.items()}
            print(f"  lambda={lam}: {vals}")

    # Experiment 6: CKA
    rand_results = load_results('exp6_cka_rand_trial*.json')
    tinit_results = load_results('exp6_cka_tinit_trial*.json')
    if rand_results:
        m = get_final_metrics(rand_results,
                              keys=['test_ts_agree', 'test_ts_kl', 'test_cka_0', 'test_cka_1', 'test_cka_2'])
        print(f"\n--- Experiment 6: CKA (Random Init) ---")
        for k, (mean, std) in m.items():
            print(f"  {k}: {mean:.3f} ± {std:.3f}")
    if tinit_results:
        m = get_final_metrics(tinit_results,
                              keys=['test_ts_agree', 'test_ts_kl', 'test_cka_0', 'test_cka_1', 'test_cka_2'])
        print(f"\n--- Experiment 6: CKA (Teacher Init) ---")
        for k, (mean, std) in m.items():
            print(f"  {k}: {mean:.3f} ± {std:.3f}")


if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print_report_values()
    plot_figure1()
    plot_figure3()
    plot_figure6a()
    plot_figure6b()
    plot_table1()
    plot_training_curves()
    plot_summary_table()
    plot_temperature_sweep()
    print(f"\nAll plots saved to {PLOTS_DIR}/")
