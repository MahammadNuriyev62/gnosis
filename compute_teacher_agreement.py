#!/usr/bin/env python3
"""
Compute pairwise agreement between independently trained teachers.
This provides a baseline: if two random models agree X%, and KD achieves Y%,
we know how much fidelity KD actually adds.
"""

import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import json
from itertools import combinations

from gnosis.models.preresnet import PreResNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = 'checkpoints'


def make_model(depth=20, num_classes=100):
    return PreResNet(num_classes=num_classes, depth=depth, input_size=32).to(DEVICE)


def get_test_loader():
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    test_dataset = torchvision.datasets.CIFAR100(
        root='data/datasets', train=False, download=True, transform=normalize)
    return DataLoader(test_dataset, batch_size=256, shuffle=False,
                      num_workers=4, pin_memory=True)


def get_predictions(model, loader):
    """Get all predictions and logits for a model."""
    model.eval()
    all_preds = []
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return (torch.cat(all_preds), torch.cat(all_logits), torch.cat(all_labels))


def compute_agreement(preds_a, preds_b):
    """Compute top-1 agreement between two sets of predictions."""
    return (preds_a == preds_b).float().mean().item() * 100


def compute_kl(logits_a, logits_b, temperature=1.0):
    """Compute KL(A || B) between two sets of logits."""
    p = F.softmax(logits_a / temperature, dim=1)
    log_q = F.log_softmax(logits_b / temperature, dim=1)
    kl = F.kl_div(log_q, p, reduction='batchmean')
    return kl.item()


def main():
    print("=" * 60)
    print("TEACHER-TEACHER PAIRWISE AGREEMENT")
    print("(Baseline for interpreting distillation fidelity)")
    print("=" * 60)

    # Load all teachers
    teachers = []
    for i in range(5):
        ckpt_path = f'{CKPT_DIR}/teacher_20_{i}.pt'
        if not os.path.exists(ckpt_path):
            print(f"Teacher {i} not found at {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = make_model()
        model.load_state_dict(ckpt['state_dict'])
        teachers.append((i, model))
        print(f"Loaded teacher {i}")

    if len(teachers) < 2:
        print("Need at least 2 teachers for pairwise comparison")
        return

    # Get test loader
    test_loader = get_test_loader()

    # Get predictions for all teachers
    print("\nComputing predictions...")
    predictions = {}
    for idx, model in teachers:
        preds, logits, labels = get_predictions(model, test_loader)
        acc = (preds == labels).float().mean().item() * 100
        predictions[idx] = {'preds': preds, 'logits': logits, 'labels': labels, 'acc': acc}
        print(f"  Teacher {idx}: {acc:.2f}% accuracy")

    # Compute pairwise agreement
    print("\n--- Pairwise Agreement (%) ---")
    agreements = []
    kl_divs = []
    for (i, _), (j, _) in combinations(teachers, 2):
        agree = compute_agreement(predictions[i]['preds'], predictions[j]['preds'])
        kl_ij = compute_kl(predictions[i]['logits'], predictions[j]['logits'])
        kl_ji = compute_kl(predictions[j]['logits'], predictions[i]['logits'])
        kl_sym = (kl_ij + kl_ji) / 2
        agreements.append(agree)
        kl_divs.append(kl_sym)
        print(f"  Teacher {i} vs Teacher {j}: {agree:.2f}% agreement, "
              f"KL={kl_sym:.4f} (KL(i||j)={kl_ij:.4f}, KL(j||i)={kl_ji:.4f})")

    mean_agree = np.mean(agreements)
    std_agree = np.std(agreements)
    mean_kl = np.mean(kl_divs)
    std_kl = np.std(kl_divs)

    print(f"\n--- Summary ---")
    print(f"Mean pairwise agreement: {mean_agree:.2f} ± {std_agree:.2f}%")
    print(f"Mean symmetric KL: {mean_kl:.4f} ± {std_kl:.4f}")
    print(f"\nFor comparison:")
    print(f"  Self-distillation T-S agreement: 71.36 ± 0.27%")
    print(f"  Self-distillation T-S KL: 0.854 ± 0.005")
    print(f"\nInterpretation:")
    print(f"  If teacher-teacher agreement ≈ distillation agreement,")
    print(f"  then KD doesn't meaningfully improve fidelity over random training.")

    # Save results
    results = {
        'pairwise_agreements': {f'{i}-{j}': agree
                                for (i, _), (j, _), agree in
                                zip(combinations(teachers, 2),
                                    combinations(teachers, 2),
                                    agreements)},
        'mean_agreement': mean_agree,
        'std_agreement': std_agree,
        'mean_kl': mean_kl,
        'std_kl': std_kl,
        'individual_accuracies': {str(idx): predictions[idx]['acc']
                                  for idx in predictions},
    }

    # Fix the zip issue - just save properly
    results['pairwise'] = []
    for (i, _), (j, _) in combinations(teachers, 2):
        agree = compute_agreement(predictions[i]['preds'], predictions[j]['preds'])
        kl_sym = (compute_kl(predictions[i]['logits'], predictions[j]['logits']) +
                  compute_kl(predictions[j]['logits'], predictions[i]['logits'])) / 2
        results['pairwise'].append({
            'teacher_i': i, 'teacher_j': j,
            'agreement': agree, 'symmetric_kl': kl_sym
        })

    os.makedirs('results', exist_ok=True)
    with open('results/teacher_pairwise_agreement.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/teacher_pairwise_agreement.json")


if __name__ == '__main__':
    main()
