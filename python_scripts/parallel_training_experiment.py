#!/usr/bin/env python3
"""
Parallelized training script for EPI model with 500 comorbidities, 800 epochs, self-connect=True.
Runs 3 repetitions for each sample size (10%, 30%, 50%, 70%, 90%) using multiprocessing.
"""

import multiprocessing as mp
import subprocess
import shlex
import time
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Configuration
SAMPLE_PCTS = [10, 30, 50, 70, 90]
SEEDS = [0, 1, 2]  # 3 repetitions
EPOCHS = 800
PYTHON_EXEC = sys.executable  # Use current Python interpreter
DATA_FILE = "EPI_heterodata_diag500_first.h5"
OFFSET = 500

def run_single_experiment(sample_pct, seed, output_dir):
    """
    Run a single training experiment.

    Args:
        sample_pct: Percentage of data to sample (10-90)
        seed: Random seed for reproducibility
        output_dir: Directory to store results

    Returns:
        dict: Experiment results including timing and success status
    """
    start_time = time.time()
    experiment_id = f"pct{sample_pct}_seed{seed}"

    # Build command
    cmd = [
        PYTHON_EXEC, "-u", "test_sample_size_500.py",
        "--file", DATA_FILE,
        "--offset", str(OFFSET),
        "--sample_pct", str(sample_pct),
        "--epochs", str(EPOCHS),
        "--self_connect_diag",
        "--seed", str(seed),
        "--device", "auto"  # Let the system choose best device
    ]

    print(f"🚀 Starting experiment: {experiment_id}")
    print(f"Command: {' '.join(cmd)}")

    try:
        # Change to EPI directory to ensure relative paths work
        original_dir = os.getcwd()
        epi_dir = "/home/zhan1/EPI"
        os.chdir(epi_dir)

        # Run the experiment
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per experiment
            cwd=epi_dir
        )

        os.chdir(original_dir)

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Completed experiment: {experiment_id} in {duration/60:.1f} min")
            return {
                'experiment_id': experiment_id,
                'sample_pct': sample_pct,
                'seed': seed,
                'success': True,
                'duration_min': duration/60,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"❌ Failed experiment: {experiment_id} (exit code: {result.returncode})")
            print(f"Error output: {result.stderr}")
            return {
                'experiment_id': experiment_id,
                'sample_pct': sample_pct,
                'seed': seed,
                'success': False,
                'duration_min': duration/60,
                'error_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
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

def analyze_results(csv_path):
    """
    Analyze the experimental results and create summary statistics.

    Args:
        csv_path: Path to the CSV file containing results

    Returns:
        pd.DataFrame: Summary statistics with mean and std for each metric
    """
    if not os.path.exists(csv_path):
        print(f"❌ Results file not found: {csv_path}")
        return None

    # Load results
    df = pd.read_csv(csv_path)

    # Filter for our experiments (500 comorbidities, 800 epochs, self_connect=1)
    experiment_df = df[
        (df['offset'] == OFFSET) &
        (df['epochs'] == EPOCHS) &
        (df['self_connect'] == 1)
    ].copy()

    if experiment_df.empty:
        print("❌ No matching experiments found in results")
        return None

    print(f"📊 Found {len(experiment_df)} experiment results")

    # Group by sample percentage and calculate statistics
    metrics = ['test_aucroc', 'test_prauc', 'test_acc', 'test_sens', 'test_spec', 'test_pos_f1']

    summary_stats = []
    for pct in SAMPLE_PCTS:
        pct_data = experiment_df[experiment_df['sample_pct'] == pct]
        if len(pct_data) == 0:
            print(f"⚠️  No data found for {pct}% sample size")
            continue

        row = {'sample_pct': pct, 'n_experiments': len(pct_data)}

        for metric in metrics:
            if metric in pct_data.columns:
                values = pct_data[metric].dropna()
                if len(values) > 0:
                    row[f'{metric}_mean'] = values.mean()
                    row[f'{metric}_std'] = values.std() if len(values) > 1 else 0.0
                    row[f'{metric}_min'] = values.min()
                    row[f'{metric}_max'] = values.max()
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
                    row[f'{metric}_min'] = np.nan
                    row[f'{metric}_max'] = np.nan

        summary_stats.append(row)

    summary_df = pd.DataFrame(summary_stats)
    return summary_df, experiment_df

def create_performance_plots(summary_df, output_dir):
    """
    Create performance plots with error bars for each evaluation metric.

    Args:
        summary_df: DataFrame with summary statistics
        output_dir: Directory to save plots
    """
    if summary_df is None or summary_df.empty:
        print("❌ No summary data available for plotting")
        return

    metrics = ['test_aucroc', 'test_prauc', 'test_acc', 'test_sens', 'test_spec', 'test_pos_f1']
    metric_labels = {
        'test_aucroc': 'AUC-ROC',
        'test_prauc': 'PR-AUC',
        'test_acc': 'Accuracy',
        'test_sens': 'Sensitivity',
        'test_spec': 'Specificity',
        'test_pos_f1': 'F1-Score'
    }

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create individual plots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Extract data for plotting
        x = summary_df['sample_pct'].values
        y = summary_df[f'{metric}_mean'].values
        yerr = summary_df[f'{metric}_std'].values

        # Remove NaN values
        mask = ~(np.isnan(y) | np.isnan(yerr))
        x_clean = x[mask]
        y_clean = y[mask]
        yerr_clean = yerr[mask]

        if len(x_clean) == 0:
            ax.text(0.5, 0.5, f'No data for {metric_labels[metric]}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric_labels[metric])
            continue

        # Create error bar plot
        ax.errorbar(x_clean, y_clean, yerr=yerr_clean,
                   marker='o', markersize=8, linewidth=2, capsize=5,
                   label=metric_labels[metric])

        # Formatting
        ax.set_xlabel('Sample Size (%)')
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(f'{metric_labels[metric]} vs Sample Size\n(Error bars: ±1 std, n=3 runs)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)

        # Set y-axis limits based on metric type
        if metric in ['test_aucroc', 'test_prauc', 'test_acc', 'test_pos_f1']:
            ax.set_ylim(0, 1)
        elif metric in ['test_sens', 'test_spec']:
            ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'performance_metrics_by_sample_size.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Performance plots saved to: {plot_path}")
    plt.close()

    # Create a combined plot showing all metrics
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(metrics)))

    for i, metric in enumerate(metrics):
        x = summary_df['sample_pct'].values
        y = summary_df[f'{metric}_mean'].values
        yerr = summary_df[f'{metric}_std'].values

        # Remove NaN values
        mask = ~(np.isnan(y) | np.isnan(yerr))
        x_clean = x[mask]
        y_clean = y[mask]
        yerr_clean = yerr[mask]

        if len(x_clean) > 0:
            ax.errorbar(x_clean, y_clean, yerr=yerr_clean,
                       marker='o', linewidth=2, capsize=3,
                       label=metric_labels[metric], color=colors[i])

    ax.set_xlabel('Sample Size (%)')
    ax.set_ylabel('Performance Score')
    ax.set_title('Model Performance vs Sample Size\n(500 Comorbidities, 800 Epochs, Self-Connect=True)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    combined_plot_path = os.path.join(output_dir, 'combined_performance_metrics.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Combined plot saved to: {combined_plot_path}")
    plt.close()

def main():
    """Main execution function."""
    start_time = time.time()

    # Setup
    epi_dir = "/home/zhan1/EPI"
    os.chdir(epi_dir)

    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)

    print("🔬 Starting Parallel EPI Model Training Experiment")
    print(f"📊 Configuration:")
    print(f"   - Sample sizes: {SAMPLE_PCTS}%")
    print(f"   - Seeds (repetitions): {SEEDS}")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Comorbidities: {OFFSET}")
    print(f"   - Self-connect: True")
    print(f"   - Total experiments: {len(SAMPLE_PCTS) * len(SEEDS)}")

    # Determine optimal number of workers
    cpu_count = mp.cpu_count()
    max_workers = min(cpu_count - 1, len(SAMPLE_PCTS) * len(SEEDS))  # Leave one CPU free
    print(f"   - CPU cores available: {cpu_count}")
    print(f"   - Parallel workers: {max_workers}")

    # Create all experiment combinations
    experiments = [(pct, seed) for pct in SAMPLE_PCTS for seed in SEEDS]

    # Run experiments in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_exp = {
            executor.submit(run_single_experiment, pct, seed, output_dir): (pct, seed)
            for pct, seed in experiments
        }

        # Collect results as they complete
        for future in as_completed(future_to_exp):
            pct, seed = future_to_exp[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"💥 Exception for experiment pct={pct}, seed={seed}: {str(e)}")
                results.append({
                    'experiment_id': f"pct{pct}_seed{seed}",
                    'sample_pct': pct,
                    'seed': seed,
                    'success': False,
                    'error': str(e)
                })

    # Summarize experiment results
    total_duration = time.time() - start_time
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful

    print(f"\n🏁 Experiment Summary:")
    print(f"   - Total time: {total_duration/60:.1f} minutes")
    print(f"   - Successful experiments: {successful}")
    print(f"   - Failed experiments: {failed}")

    # Save experiment log
    results_df = pd.DataFrame(results)
    log_path = os.path.join(output_dir, 'experiment_log.csv')
    results_df.to_csv(log_path, index=False)
    print(f"📝 Experiment log saved to: {log_path}")

    # Analyze results from the main CSV file
    csv_path = os.path.join(epi_dir, 'sample_size_performance.csv')
    summary_df, experiment_df = analyze_results(csv_path)

    if summary_df is not None:
        # Save summary statistics
        summary_path = os.path.join(output_dir, 'performance_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"📊 Performance summary saved to: {summary_path}")

        # Create plots
        create_performance_plots(summary_df, output_dir)

        # Print summary table
        print(f"\n📈 Performance Summary:")
        print("=" * 80)
        for _, row in summary_df.iterrows():
            pct = row['sample_pct']
            n = row['n_experiments']
            print(f"\nSample Size: {pct}% (n={n} experiments)")
            print("-" * 40)

            metrics = ['test_aucroc', 'test_prauc', 'test_acc', 'test_sens', 'test_spec', 'test_pos_f1']
            for metric in metrics:
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'
                if mean_col in row and not pd.isna(row[mean_col]):
                    mean_val = row[mean_col]
                    std_val = row[std_col] if not pd.isna(row[std_col]) else 0.0
                    metric_name = metric.replace('test_', '').upper()
                    print(f"  {metric_name:8}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"\n✨ Experiment completed! Results saved in: {output_dir}")

if __name__ == '__main__':
    main()