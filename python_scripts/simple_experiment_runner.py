#!/usr/bin/env python3
"""
Simple experiment runner for EPI model with 500 comorbidities, 800 epochs, self-connect=True.
Runs experiments sequentially and then analyzes results.
"""

import subprocess
import time
import os
import sys
import csv
from pathlib import Path

# Configuration
SAMPLE_PCTS = [10, 30, 50, 70, 90]
SEEDS = [0, 1, 2]  # 3 repetitions
EPOCHS = 800
PYTHON_EXEC = sys.executable
DATA_FILE = "EPI_heterodata_diag500_first.h5"
OFFSET = 500

def run_single_experiment(sample_pct, seed):
    """
    Run a single training experiment.

    Args:
        sample_pct: Percentage of data to sample (10-90)
        seed: Random seed for reproducibility

    Returns:
        dict: Experiment results including timing and success status
    """
    start_time = time.time()
    experiment_id = f"pct{sample_pct}_seed{seed}"

    # Build command
    cmd = [
        PYTHON_EXEC, "-u", "python_scripts/test_sample_size_500.py",
        "--file", f"data/{DATA_FILE}",
        "--offset", str(OFFSET),
        "--sample_pct", str(sample_pct),
        "--epochs", str(EPOCHS),
        "--self_connect_diag",
        "--seed", str(seed),
        "--device", "auto"
    ]

    print(f"🚀 Starting experiment: {experiment_id}")
    print(f"Command: {' '.join(cmd)}")

    try:
        # Run the experiment
        result = subprocess.run(
            cmd,
            cwd="/home/zhan1/EPI",
            timeout=3600,  # 1 hour timeout per experiment
            text=True,
            capture_output=True
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Completed experiment: {experiment_id} in {duration/60:.1f} min")
            return {
                'experiment_id': experiment_id,
                'sample_pct': sample_pct,
                'seed': seed,
                'success': True,
                'duration_min': duration/60,
                'stdout': result.stdout[-1000:],  # Last 1000 chars
                'stderr': result.stderr[-1000:] if result.stderr else ""
            }
        else:
            print(f"❌ Failed experiment: {experiment_id} (exit code: {result.returncode})")
            print(f"Error: {result.stderr}")
            return {
                'experiment_id': experiment_id,
                'sample_pct': sample_pct,
                'seed': seed,
                'success': False,
                'duration_min': duration/60,
                'error_code': result.returncode,
                'stdout': result.stdout[-1000:] if result.stdout else "",
                'stderr': result.stderr[-1000:] if result.stderr else ""
            }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ Timeout experiment: {experiment_id} after {duration/60:.1f} min")
        return {
            'experiment_id': experiment_id,
            'sample_pct': sample_pct,
            'seed': seed,
            'success': False,
            'duration_min': duration/60,
            'error': 'timeout'
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 Exception in experiment: {experiment_id}: {str(e)}")
        return {
            'experiment_id': experiment_id,
            'sample_pct': sample_pct,
            'seed': seed,
            'success': False,
            'duration_min': duration/60,
            'error': str(e)
        }

def analyze_csv_results(csv_path):
    """
    Simple analysis of CSV results without pandas.

    Args:
        csv_path: Path to the CSV file

    Returns:
        dict: Summary statistics
    """
    if not os.path.exists(csv_path):
        print(f"❌ Results file not found: {csv_path}")
        return None

    # Read CSV manually
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter for our experiments
            if (int(row.get('offset', 0)) == OFFSET and
                int(row.get('epochs', 0)) == EPOCHS and
                int(row.get('self_connect', 0)) == 1):
                results.append(row)

    if not results:
        print("❌ No matching experiments found")
        return None

    print(f"📊 Found {len(results)} experiment results")

    # Group by sample percentage
    metrics = ['test_aucroc', 'test_prauc', 'test_acc', 'test_sens', 'test_spec', 'test_pos_f1']
    summary = {}

    for pct in SAMPLE_PCTS:
        pct_results = [r for r in results if int(r.get('sample_pct', 0)) == pct]
        if not pct_results:
            continue

        pct_stats = {'sample_pct': pct, 'n_experiments': len(pct_results)}

        for metric in metrics:
            values = []
            for result in pct_results:
                try:
                    val = float(result.get(metric, 0))
                    values.append(val)
                except (ValueError, TypeError):
                    continue

            if values:
                mean_val = sum(values) / len(values)
                if len(values) > 1:
                    # Calculate standard deviation manually
                    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
                    std_val = variance ** 0.5
                else:
                    std_val = 0.0

                pct_stats[f'{metric}_mean'] = mean_val
                pct_stats[f'{metric}_std'] = std_val
                pct_stats[f'{metric}_values'] = values

        summary[pct] = pct_stats

    return summary

def create_simple_plots():
    """
    Create simple plots using matplotlib if available.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        csv_path = "/home/zhan1/EPI/sample_size_performance.csv"
        summary = analyze_csv_results(csv_path)

        if not summary:
            return

        # Create plots
        metrics = ['test_aucroc', 'test_prauc', 'test_acc', 'test_sens', 'test_spec', 'test_pos_f1']
        metric_labels = {
            'test_aucroc': 'AUC-ROC',
            'test_prauc': 'PR-AUC',
            'test_acc': 'Accuracy',
            'test_sens': 'Sensitivity',
            'test_spec': 'Specificity',
            'test_pos_f1': 'F1-Score'
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Extract data
            x_vals = []
            y_vals = []
            err_vals = []

            for pct in sorted(summary.keys()):
                if f'{metric}_mean' in summary[pct]:
                    x_vals.append(pct)
                    y_vals.append(summary[pct][f'{metric}_mean'])
                    err_vals.append(summary[pct][f'{metric}_std'])

            if x_vals:
                ax.errorbar(x_vals, y_vals, yerr=err_vals,
                           marker='o', linewidth=2, capsize=5)
                ax.set_xlabel('Sample Size (%)')
                ax.set_ylabel(metric_labels[metric])
                ax.set_title(f'{metric_labels[metric]} vs Sample Size')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig('/home/zhan1/EPI/experiment_results.png', dpi=300, bbox_inches='tight')
        print("📈 Plot saved to experiment_results.png")
        plt.close()

    except ImportError:
        print("⚠️  matplotlib not available, skipping plots")

def main():
    """Main execution function."""
    start_time = time.time()

    # Setup
    os.chdir("/home/zhan1/EPI")

    print("🔬 Starting EPI Model Training Experiment")
    print(f"📊 Configuration:")
    print(f"   - Sample sizes: {SAMPLE_PCTS}%")
    print(f"   - Seeds (repetitions): {SEEDS}")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Comorbidities: {OFFSET}")
    print(f"   - Self-connect: True")
    print(f"   - Total experiments: {len(SAMPLE_PCTS) * len(SEEDS)}")

    # Create experiment list
    experiments = [(pct, seed) for pct in SAMPLE_PCTS for seed in SEEDS]

    # Run experiments sequentially
    results = []
    for i, (pct, seed) in enumerate(experiments, 1):
        print(f"\n📍 Running experiment {i}/{len(experiments)}")
        result = run_single_experiment(pct, seed)
        results.append(result)

        # Brief status update
        successful = sum(1 for r in results if r.get('success', False))
        print(f"   Progress: {successful}/{i} successful")

    # Final summary
    total_duration = time.time() - start_time
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful

    print(f"\n🏁 Experiment Summary:")
    print(f"   - Total time: {total_duration/60:.1f} minutes")
    print(f"   - Successful experiments: {successful}")
    print(f"   - Failed experiments: {failed}")

    # Analyze results
    print(f"\n📊 Analyzing results...")
    csv_path = "/home/zhan1/EPI/sample_size_performance.csv"
    summary = analyze_csv_results(csv_path)

    if summary:
        print(f"\n📈 Performance Summary:")
        print("=" * 60)

        for pct in sorted(summary.keys()):
            stats = summary[pct]
            n = stats['n_experiments']
            print(f"\nSample Size: {pct}% (n={n} experiments)")
            print("-" * 30)

            metrics = ['test_aucroc', 'test_prauc', 'test_acc', 'test_sens', 'test_spec', 'test_pos_f1']
            for metric in metrics:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                if mean_key in stats:
                    metric_name = metric.replace('test_', '').upper()
                    print(f"  {metric_name:8}: {stats[mean_key]:.4f} ± {stats[std_key]:.4f}")

    # Create plots
    create_simple_plots()

    print(f"\n✨ Experiment completed!")

if __name__ == '__main__':
    main()