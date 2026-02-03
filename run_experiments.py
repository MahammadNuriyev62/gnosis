#!/usr/bin/env python3
"""
Replication experiments for "Does Knowledge Distillation Really Work?"
(Stanton et al., NeurIPS 2021)

Runs all key experiments from the paper on CIFAR-100 with PreResNet-20.
"""

import os
import sys
import json
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter

# Project imports
from gnosis.models.preresnet import PreResNet, freeze_batchnorm
from gnosis.models.ensemble import ClassifierEnsemble, ClassifierEnsembleLoss
from gnosis.distillation.classification import (
    ClassifierTeacherLoss,
    ClassifierStudentLoss,
    TeacherStudentFwdCrossEntLoss,
    reduce_ensemble_logits,
)
from gnosis.distillation.dataloaders import DistillLoader
from gnosis.boilerplate.supervised_epoch import supervised_epoch
from gnosis.boilerplate.distillation_epoch import distillation_epoch
from gnosis.boilerplate.eval_epoch import eval_epoch
from gnosis.utils.initialization import interpolate_net
from gnosis.utils.metrics import preact_cka
from cka.CKA import linear_CKA, kernel_CKA

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = 'results'
CKPT_DIR = 'checkpoints'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def try_cuda(*args):
    return [a.to(DEVICE) if hasattr(a, 'to') else a for a in args] if len(args) > 1 else args[0].to(DEVICE)


def get_cifar100_loaders(batch_size=128, augment=True, normalization='unitcube'):
    if normalization == 'unitcube':
        # Map [0, 1] -> [-1, 1] to match original paper's unitcube normalization
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    else:
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            normalize,
        ])
    else:
        train_transform = normalize

    test_transform = normalize

    train_dataset = torchvision.datasets.CIFAR100(
        root='data/datasets', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR100(
        root='data/datasets', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, test_loader, train_dataset, test_dataset


def make_model(depth=20, num_classes=100):
    return PreResNet(num_classes=num_classes, depth=depth, input_size=32).to(DEVICE)


def train_teacher(depth=20, num_classes=100, num_epochs=200, lr=0.1,
                  batch_size=256, seed=0, save_path=None):
    """Train a single teacher model on CIFAR-100."""
    set_seed(seed)
    train_loader, test_loader, _, _ = get_cifar100_loaders(batch_size=batch_size)

    model = make_model(depth=depth, num_classes=num_classes)
    loss_fn = ClassifierTeacherLoss(model)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=1e-4, nesterov=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.)

    records = []
    for epoch in range(1, num_epochs + 1):
        train_metrics = supervised_epoch(model, train_loader, optimizer, lr_scheduler,
                                         epoch=epoch, loss_fn=loss_fn)
        if epoch % 10 == 0 or epoch == num_epochs:
            eval_metrics = eval_epoch(model, test_loader, epoch=epoch, loss_fn=loss_fn)
            records.append({**train_metrics, **eval_metrics})
            print(f"  Teacher (seed={seed}) epoch {epoch}: "
                  f"train_acc={train_metrics['train_acc']:.2f}%, "
                  f"test_acc={eval_metrics['test_acc']:.2f}%")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'state_dict': model.state_dict(),
            'records': records,
            'seed': seed,
            'depth': depth,
        }, save_path)
        print(f"  Saved teacher to {save_path}")

    return model, records


def load_or_train_teachers(num_teachers=5, depth=20, num_epochs=200):
    """Load existing teacher checkpoints or train new ones."""
    teachers = []
    all_records = []
    for i in range(num_teachers):
        ckpt_path = f'{CKPT_DIR}/teacher_{depth}_{i}.pt'
        if os.path.exists(ckpt_path):
            print(f"Loading teacher {i} from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model = make_model(depth=depth)
            model.load_state_dict(ckpt['state_dict'])
            teachers.append(model)
            all_records.append(ckpt['records'])
        else:
            print(f"Training teacher {i}...")
            model, records = train_teacher(depth=depth, num_epochs=num_epochs,
                                           seed=i, save_path=ckpt_path)
            teachers.append(model)
            all_records.append(records)
    return teachers, all_records


def distill_student(teachers, depth=20, num_classes=100, num_epochs=300,
                    lr=5e-2, batch_size=128, temp=4.0, alpha=0.0,
                    seed=0, mixup_alpha=0.0, augment_transforms=None,
                    init_type=None, init_lambda=0.0, freeze_bn=False,
                    optimizer_type='sgd', weight_decay=1e-4, compute_cka=False):
    """Run knowledge distillation from teacher(s) to a student."""
    set_seed(seed)

    # Build data loaders
    train_loader, test_loader, train_dataset, _ = get_cifar100_loaders(batch_size=batch_size)

    # Build teacher ensemble
    teacher = ClassifierEnsemble(*teachers)
    teacher.eval()

    # Build distill loader
    distill_loader = DistillLoader(
        teacher=teacher,
        datasets=[train_dataset],
        temp=temp,
        mixup_alpha=mixup_alpha,
        mixup_portion=1.0,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        synth_ratio=0.0,
    )

    # Build student
    student = make_model(depth=depth, num_classes=num_classes)

    # Student initialization experiments
    if init_type == 'final' and len(teachers) == 1:
        # init_lambda follows paper convention: 0=random, 1=teacher
        # interpolate_net uses distance_ratio: 0=teacher, 1=random
        distance_ratio = 1.0 - init_lambda
        print(f"  Initializing student near final teacher weights (lambda={init_lambda}, dist_ratio={distance_ratio})")
        student = interpolate_net(student, teachers[0].state_dict(),
                                  distance_ratio, train_loader, freeze_bn)
        lr = max(lr * distance_ratio, 1e-6)
    elif init_type == 'init':
        print(f"  Initializing student near initial teacher weights (lambda={init_lambda})")
        # This would require saved init checkpoints; skip if not available

    # Build loss
    base_loss = TeacherStudentFwdCrossEntLoss()
    student_loss = ClassifierStudentLoss(student, base_loss, alpha)

    # Build optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(student.parameters(), lr=lr * 0.1)
    else:
        optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9,
                              weight_decay=weight_decay, nesterov=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Evaluate teacher on test set
    teacher_loss_fn = ClassifierEnsembleLoss(teacher)
    teacher_test_metrics = eval_epoch(teacher, test_loader, epoch=0, loss_fn=teacher_loss_fn)
    print(f"  Teacher test acc: {teacher_test_metrics['test_acc']:.2f}%")

    # Training loop
    records = []
    eval_metrics = eval_epoch(student, test_loader, epoch=0,
                              loss_fn=student_loss, teacher=teacher)
    records.append(eval_metrics)

    for epoch in range(1, num_epochs + 1):
        train_metrics = distillation_epoch(student, distill_loader, optimizer,
                                           lr_scheduler, epoch=epoch,
                                           loss_fn=student_loss, freeze_bn=freeze_bn)
        if epoch % 10 == 0 or epoch == num_epochs:
            eval_metrics = eval_epoch(student, test_loader, epoch=epoch,
                                      loss_fn=student_loss, teacher=teacher,
                                      with_cka=compute_cka)
            metrics = {**train_metrics, **eval_metrics}
            records.append(metrics)
            print(f"  Student epoch {epoch}: "
                  f"train_acc={train_metrics['train_acc']:.2f}%, "
                  f"test_acc={eval_metrics['test_acc']:.2f}%, "
                  f"test_agree={eval_metrics.get('test_ts_agree', 'N/A')}%, "
                  f"test_kl={eval_metrics.get('test_ts_kl', 'N/A')}")

    for r in records:
        r['teacher_test_acc'] = teacher_test_metrics['test_acc']

    return student, records


def save_checkpoint(model, name):
    """Save a model checkpoint."""
    os.makedirs('checkpoints', exist_ok=True)
    path = f'checkpoints/{name}.pt'
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}")


def save_results(records, name, student=None):
    """Save experiment results to JSON and optionally save model checkpoint."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if student is not None:
        save_checkpoint(student, name)
    path = f'{RESULTS_DIR}/{name}.json'
    # Convert any non-serializable values
    clean_records = []
    for r in records:
        clean = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            elif isinstance(v, torch.Tensor):
                clean[k] = v.item()
            else:
                clean[k] = v
        clean_records.append(clean)
    with open(path, 'w') as f:
        json.dump(clean_records, f, indent=2)
    print(f"Saved results to {path}")


# ============================================================
# EXPERIMENTS
# ============================================================

def experiment_1_self_distillation(num_trials=3):
    """
    Replicate Figure 1(a): Self-distillation (1 teacher -> 1 student).
    Shows that student outperforms teacher in accuracy but has poor fidelity.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Self-distillation (Figure 1a)")
    print("=" * 70)

    all_results = {}
    for trial in range(num_trials):
        print(f"\n--- Trial {trial} ---")
        teachers, _ = load_or_train_teachers(num_teachers=1, depth=20, num_epochs=200)
        student, records = distill_student(
            teachers=teachers, depth=20, num_epochs=300,
            temp=4.0, alpha=0.0, seed=trial + 100,
        )
        save_results(records, f'exp1_self_distill_trial{trial}', student=student)
        all_results[trial] = records

    return all_results


def experiment_2_ensemble_distillation(num_trials=3):
    """
    Replicate Figure 1(b): Ensemble distillation (3 teachers -> 1 student).
    Shows that fidelity becomes positively correlated with generalization.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Ensemble distillation (Figure 1b)")
    print("=" * 70)

    all_results = {}
    for num_teachers in [3, 5]:
        teachers, _ = load_or_train_teachers(num_teachers=num_teachers, depth=20, num_epochs=200)
        for trial in range(num_trials):
            result_file = f'{RESULTS_DIR}/exp2_ensemble{num_teachers}_trial{trial}.json'
            if os.path.exists(result_file):
                print(f"\n--- {num_teachers} teachers, Trial {trial} --- SKIPPING (results exist)")
                continue
            print(f"\n--- {num_teachers} teachers, Trial {trial} ---")
            student, records = distill_student(
                teachers=teachers, depth=20, num_epochs=300,
                temp=4.0, alpha=0.0, seed=trial + 200,
            )
            save_results(records, f'exp2_ensemble{num_teachers}_trial{trial}', student=student)
            all_results[f'{num_teachers}t_trial{trial}'] = records

    return all_results


def experiment_3_augmentation(num_trials=3):
    """
    Replicate Figure 3: Effect of data augmentation on fidelity vs accuracy.
    Tests: Baseline(tau=1), Baseline(tau=4), MixUp(tau=4).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Data augmentation effects (Figure 3)")
    print("=" * 70)

    teachers, _ = load_or_train_teachers(num_teachers=5, depth=20, num_epochs=200)
    all_results = {}

    configs = [
        ('baseline_t1', {'temp': 1.0, 'mixup_alpha': 0.0}),
        ('baseline_t4', {'temp': 4.0, 'mixup_alpha': 0.0}),
        ('mixup_t4', {'temp': 4.0, 'mixup_alpha': 1.0}),
    ]

    for name, kwargs in configs:
        for trial in range(num_trials):
            result_file = f'{RESULTS_DIR}/exp3_{name}_trial{trial}.json'
            if os.path.exists(result_file):
                print(f"\n--- {name}, Trial {trial} --- SKIPPING (results exist)")
                continue
            print(f"\n--- {name}, Trial {trial} ---")
            student, records = distill_student(
                teachers=teachers, depth=20, num_epochs=300,
                alpha=0.0, seed=trial + 300, **kwargs,
            )
            save_results(records, f'exp3_{name}_trial{trial}', student=student)
            all_results[f'{name}_trial{trial}'] = records

    return all_results


def experiment_4_optimization(num_trials=3):
    """
    Replicate Figure 6(a): SGD vs Adam, different epoch counts.
    Self-distillation with PreResNet-20 on CIFAR-100.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Optimization effects (Figure 6a)")
    print("=" * 70)

    teachers, _ = load_or_train_teachers(num_teachers=1, depth=20, num_epochs=200)
    all_results = {}

    configs = [
        ('sgd_300ep', {'optimizer_type': 'sgd', 'num_epochs': 300, 'weight_decay': 0}),
        ('sgd_600ep', {'optimizer_type': 'sgd', 'num_epochs': 600, 'weight_decay': 0}),
        ('adam_300ep', {'optimizer_type': 'adam', 'num_epochs': 300, 'weight_decay': 0}),
        ('adam_600ep', {'optimizer_type': 'adam', 'num_epochs': 600, 'weight_decay': 0}),
    ]

    for name, kwargs in configs:
        for trial in range(num_trials):
            result_file = f'{RESULTS_DIR}/exp4_{name}_trial{trial}.json'
            if os.path.exists(result_file):
                print(f"\n--- {name}, Trial {trial} --- SKIPPING (results exist)")
                continue
            print(f"\n--- {name}, Trial {trial} ---")
            student, records = distill_student(
                teachers=teachers, depth=20,
                temp=4.0, alpha=0.0, seed=trial + 400, **kwargs,
            )
            save_results(records, f'exp4_{name}_trial{trial}', student=student)
            all_results[f'{name}_trial{trial}'] = records

    return all_results


def experiment_5_initialization(num_trials=3):
    """
    Replicate Figure 6(b): Student initialization near teacher weights.
    Self-distillation with varying lambda.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Initialization proximity (Figure 6b)")
    print("=" * 70)

    teachers, _ = load_or_train_teachers(num_teachers=1, depth=20, num_epochs=200)
    all_results = {}

    lambdas = [0.0, 0.25, 0.375, 0.5, 1.0]

    for lam in lambdas:
        for trial in range(num_trials):
            result_file = f'{RESULTS_DIR}/exp5_init_lam{lam}_trial{trial}.json'
            if os.path.exists(result_file):
                print(f"\n--- lambda={lam}, Trial {trial} --- SKIPPING (results exist)")
                continue
            print(f"\n--- lambda={lam}, Trial {trial} ---")
            student, records = distill_student(
                teachers=teachers, depth=20, num_epochs=300,
                temp=4.0, alpha=0.0, seed=trial + 500,
                init_type='final', init_lambda=lam,
            )
            save_results(records, f'exp5_init_lam{lam}_trial{trial}', student=student)
            all_results[f'lam{lam}_trial{trial}'] = records

    return all_results


def experiment_6_cka_analysis(num_trials=3):
    """
    Replicate Table 1: CKA analysis with random vs teacher initialization.
    Self-distillation with PreResNet-20 on CIFAR-100.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: CKA Analysis (Table 1)")
    print("=" * 70)

    # We need teacher init checkpoints - save them during training
    all_results = {}

    for trial in range(num_trials):
        rand_file = f'{RESULTS_DIR}/exp6_cka_rand_trial{trial}.json'
        tinit_file = f'{RESULTS_DIR}/exp6_cka_tinit_trial{trial}.json'
        if os.path.exists(rand_file) and os.path.exists(tinit_file):
            print(f"\n--- Trial {trial} --- SKIPPING (results exist)")
            continue
        print(f"\n--- Trial {trial} ---")
        set_seed(trial)

        train_loader, test_loader, train_dataset, _ = get_cifar100_loaders(batch_size=128)

        # Train a teacher, saving initial weights
        teacher_model = make_model(depth=20)
        teacher_init_state = copy.deepcopy(teacher_model.state_dict())
        teacher_loss = ClassifierTeacherLoss(teacher_model)

        optimizer = optim.SGD(teacher_model.parameters(), lr=0.1, momentum=0.9,
                              weight_decay=1e-4, nesterov=True)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.)

        for epoch in range(1, 201):
            supervised_epoch(teacher_model, train_loader, optimizer, lr_scheduler,
                             epoch=epoch, loss_fn=teacher_loss)
            if epoch % 50 == 0:
                eval_metrics = eval_epoch(teacher_model, test_loader, epoch=epoch,
                                          loss_fn=teacher_loss)
                print(f"  Teacher epoch {epoch}: test_acc={eval_metrics['test_acc']:.2f}%")

        teacher_model.eval()

        # Distill with RANDOM init
        print(f"  Distilling with random init...")
        set_seed(trial + 1000)
        student_rand, records_rand = distill_student(
            teachers=[teacher_model], depth=20, num_epochs=300,
            temp=4.0, alpha=0.0, seed=trial + 1000, compute_cka=True,
        )
        save_results(records_rand, f'exp6_cka_rand_trial{trial}', student=student_rand)

        # Distill with TEACHER init (lambda=0, i.e. init at teacher init weights)
        print(f"  Distilling with teacher-init...")
        set_seed(trial + 2000)
        student_tinit = make_model(depth=20)
        student_tinit.load_state_dict(teacher_init_state)

        teacher_ens = ClassifierEnsemble(teacher_model)
        teacher_ens.eval()
        distill_loader = DistillLoader(
            teacher=teacher_ens, datasets=[train_dataset], temp=4.0,
            mixup_alpha=0., mixup_portion=1., batch_size=128,
            shuffle=True, drop_last=False, synth_ratio=0.,
        )
        base_loss = TeacherStudentFwdCrossEntLoss()
        student_loss = ClassifierStudentLoss(student_tinit, base_loss, alpha=0.0)

        opt2 = optim.SGD(student_tinit.parameters(), lr=5e-2, momentum=0.9,
                         weight_decay=1e-4, nesterov=True)
        sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=300, eta_min=1e-6)

        records_tinit = []
        for epoch in range(1, 301):
            train_metrics = distillation_epoch(student_tinit, distill_loader, opt2,
                                               sched2, epoch=epoch, loss_fn=student_loss)
            if epoch % 10 == 0 or epoch == 300:
                eval_metrics = eval_epoch(student_tinit, test_loader, epoch=epoch,
                                          loss_fn=student_loss, teacher=teacher_ens, with_cka=True)
                records_tinit.append({**train_metrics, **eval_metrics})
                if epoch % 50 == 0:
                    print(f"  Student (teacher-init) epoch {epoch}: "
                          f"test_acc={eval_metrics['test_acc']:.2f}%, "
                          f"agree={eval_metrics.get('test_ts_agree', 'N/A')}")

        save_results(records_tinit, f'exp6_cka_tinit_trial{trial}', student=student_tinit)
        all_results[f'trial{trial}'] = {
            'random_init': records_rand,
            'teacher_init': records_tinit,
        }

    return all_results


def experiment_temperature_sweep(num_trials=3):
    """
    Additional experiment: Effect of temperature on fidelity.
    Self-distillation with varying temperatures.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Temperature sweep")
    print("=" * 70)

    teachers, _ = load_or_train_teachers(num_teachers=1, depth=20, num_epochs=200)
    all_results = {}

    for temp in [1.0, 2.0, 4.0, 8.0, 16.0]:
        for trial in range(num_trials):
            print(f"\n--- T={temp}, Trial {trial} ---")
            student, records = distill_student(
                teachers=teachers, depth=20, num_epochs=300,
                temp=temp, alpha=0.0, seed=trial + 700,
            )
            save_results(records, f'exp_temp{temp}_trial{trial}', student=student)
            all_results[f'T{temp}_trial{trial}'] = records

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', '1', '2', '3', '4', '5', '6', 'temp',
                                 'teachers'],
                        help='Which experiment to run')
    parser.add_argument('--num_trials', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    if args.exp in ['all', 'teachers']:
        print("\n" + "=" * 70)
        print("TRAINING TEACHER MODELS")
        print("=" * 70)
        teachers, teacher_records = load_or_train_teachers(
            num_teachers=5, depth=20, num_epochs=200)
        for i, recs in enumerate(teacher_records):
            save_results(recs, f'teacher_{i}_records')

    if args.exp in ['all', '1']:
        experiment_1_self_distillation(num_trials=args.num_trials)

    if args.exp in ['all', '2']:
        experiment_2_ensemble_distillation(num_trials=args.num_trials)

    if args.exp in ['all', '3']:
        experiment_3_augmentation(num_trials=args.num_trials)

    if args.exp in ['all', '4']:
        experiment_4_optimization(num_trials=args.num_trials)

    if args.exp in ['all', '5']:
        experiment_5_initialization(num_trials=args.num_trials)

    if args.exp in ['all', '6']:
        experiment_6_cka_analysis(num_trials=args.num_trials)

    if args.exp == 'temp':
        experiment_temperature_sweep(num_trials=args.num_trials)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
