"""
Generate two separate PNG figures:
  1. loss_curves.png       - Train Loss vs Validation Loss
  2. objective_curves.png  - Validation Objective

Usage:
    python plot_training.py [--run_dir RUN_DIR] [--output OUTPUT_DIR]
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_history(run_dir):
    """Load epoch_history.jsonl as a list of dicts."""
    history_file = os.path.join(run_dir, 'epoch_history.jsonl')
    if not os.path.isfile(history_file):
        raise FileNotFoundError(f"No epoch_history.jsonl found in {run_dir}")
    records = []
    with open(history_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def plot_loss_separate(records, output_dir):
    """
    Figure 1: Train Loss vs Validation Loss.
    """
    epochs = [r['epoch'] for r in records]
    train_loss = [r['train_loss'] for r in records]
    val_loss = [r['val_loss'] for r in records]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Figure 1 saved to {out_path}")
    plt.close(fig)

def plot_objective_separate(records, output_dir):
    """
    Figure 2: Validation Objective.
    """
    epochs = [r['epoch'] for r in records]
    val_obj = [r['val_objective'] for r in records]

    # Best epoch info
    best_idx = min(range(len(records)), key=lambda i: records[i]['best_val_objective'])
    best_epoch = records[best_idx]['epoch']
    best_value = records[best_idx]['best_val_objective']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, val_obj, 'g-', label='Validation Objective', linewidth=2)
    ax.axvline(x=best_epoch, color='gray', linestyle=':', alpha=0.7,
               label=f'Best epoch = {best_epoch} (val={best_value:.4f})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Objective', fontsize=12)
    ax.set_title('Validation Objective Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    out_path = os.path.join(output_dir, 'objective_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Figure 2 saved to {out_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Generate two separate training curve figures.')
    parser.add_argument('--run_dir', type=str, default=os.getcwd(),
                        help='Directory containing epoch_history.jsonl (default: current dir)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots (default: same as run_dir)')
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        print(f"Error: Directory {run_dir} does not exist.")
        return 1

    output_dir = args.output if args.output else run_dir
    os.makedirs(output_dir, exist_ok=True)

    try:
        records = load_history(run_dir)
        print(f"Loaded {len(records)} epochs from {run_dir}")
    except FileNotFoundError as e:
        print(e)
        return 1

    plot_loss_separate(records, output_dir)
    plot_objective_separate(records, output_dir)

    print("All plots generated successfully.")
    return 0

if __name__ == '__main__':
    exit(main())