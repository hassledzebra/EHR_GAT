#!/usr/bin/env python3
"""
Create separate training and testing performance plots to show the comparison.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import csv
from progress_tracker import ProgressTracker

def load_and_analyze_train_test_data():
    """Load data and separate training vs testing metrics."""
    tracker = ProgressTracker()
    tracker.log_message("📊 Analyzing training vs testing performance...")

    csv_path = "/home/zhan1/EPI/sample_size_performance.csv"
    SAMPLE_PCTS = [10, 30, 50, 70, 90, 100]

    # Read results
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (int(row.get('offset', 0)) == 500 and
                int(row.get('epochs', 0)) == 800 and
                int(row.get('self_connect', 0)) == 1):
                results.append(row)

    # Separate training and testing metrics
    train_metrics = ['train_aucroc', 'train_prauc', 'train_acc', 'train_sens', 'train_spec', 'train_pos_f1']
    test_metrics = ['test_aucroc', 'test_prauc', 'test_acc', 'test_sens', 'test_spec', 'test_pos_f1']

    train_summary = {}
    test_summary = {}

    for pct in SAMPLE_PCTS:
        pct_results = [r for r in results if int(r.get('sample_pct', 0)) == pct]
        if not pct_results:
            continue

        # Training metrics
        train_stats = {'sample_pct': pct, 'n_experiments': len(pct_results)}
        for metric in train_metrics:
            values = []
            for result in pct_results:
                try:
                    val = float(result.get(metric, 0))
                    values.append(val)
                except (ValueError, TypeError):
                    continue
            if values:
                values = np.array(values)
                train_stats[f'{metric}_mean'] = float(values.mean())
                train_stats[f'{metric}_std'] = float(values.std()) if len(values) > 1 else 0.0
        train_summary[pct] = train_stats

        # Testing metrics
        test_stats = {'sample_pct': pct, 'n_experiments': len(pct_results)}
        for metric in test_metrics:
            values = []
            for result in pct_results:
                try:
                    val = float(result.get(metric, 0))
                    values.append(val)
                except (ValueError, TypeError):
                    continue
            if values:
                values = np.array(values)
                test_stats[f'{metric}_mean'] = float(values.mean())
                test_stats[f'{metric}_std'] = float(values.std()) if len(values) > 1 else 0.0
        test_summary[pct] = test_stats

    return train_summary, test_summary

def create_train_test_comparison_plots(train_summary, test_summary):
    """Create comprehensive training vs testing comparison plots."""
    tracker = ProgressTracker()
    tracker.log_message("📊 Creating training vs testing comparison plots...")

    # Define metrics and labels
    base_metrics = ['aucroc', 'prauc', 'acc', 'sens', 'spec', 'pos_f1']
    metric_labels = {
        'aucroc': 'AUC-ROC',
        'prauc': 'PR-AUC',
        'acc': 'Accuracy',
        'sens': 'Sensitivity',
        'spec': 'Specificity',
        'pos_f1': 'F1-Score'
    }

    # Set style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # 1. Individual metric comparison plots (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()

    colors = ['steelblue', 'darkorange', 'green', 'red', 'purple', 'brown']

    for i, metric in enumerate(base_metrics):
        ax = axes[i]

        # Extract training data
        train_x, train_y, train_err = [], [], []
        for pct in sorted(train_summary.keys()):
            train_metric = f'train_{metric}'
            if f'{train_metric}_mean' in train_summary[pct]:
                train_x.append(pct)
                train_y.append(train_summary[pct][f'{train_metric}_mean'])
                train_err.append(train_summary[pct][f'{train_metric}_std'])

        # Extract testing data
        test_x, test_y, test_err = [], [], []
        for pct in sorted(test_summary.keys()):
            test_metric = f'test_{metric}'
            if f'{test_metric}_mean' in test_summary[pct]:
                test_x.append(pct)
                test_y.append(test_summary[pct][f'{test_metric}_mean'])
                test_err.append(test_summary[pct][f'{test_metric}_std'])

        # Plot training performance
        if train_x:
            ax.errorbar(train_x, train_y, yerr=train_err,
                       marker='o', markersize=8, linewidth=2.5, capsize=6,
                       color=colors[i], alpha=0.8, label='Training',
                       markerfacecolor='white', markeredgewidth=2)

        # Plot testing performance
        if test_x:
            ax.errorbar(test_x, test_y, yerr=test_err,
                       marker='s', markersize=8, linewidth=2.5, capsize=6,
                       color=colors[i], alpha=0.6, label='Testing', linestyle='--',
                       markerfacecolor='white', markeredgewidth=2)

        ax.set_xlabel('Sample Size (%)', fontsize=12)
        ax.set_ylabel(metric_labels[metric], fontsize=12)
        ax.set_title(f'{metric_labels[metric]} - Training vs Testing\n(500 comorbidities, 800 epochs, n=3 runs)',
                    fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xlim(5, 105)
        ax.set_ylim(0, 1)

    plt.suptitle('Training vs Testing Performance Comparison\nEPI Model with Self-Connect=True',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    individual_plot = "/home/zhan1/EPI/train_test_individual_metrics.png"
    plt.savefig(individual_plot, dpi=300, bbox_inches='tight', facecolor='white')
    tracker.log_message(f"📈 Individual train/test plots saved to: {individual_plot}")
    plt.close()

    # 2. Combined training vs testing plot (all metrics)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Training performance plot
    for i, metric in enumerate(base_metrics):
        train_x, train_y, train_err = [], [], []
        for pct in sorted(train_summary.keys()):
            train_metric = f'train_{metric}'
            if f'{train_metric}_mean' in train_summary[pct]:
                train_x.append(pct)
                train_y.append(train_summary[pct][f'{train_metric}_mean'])
                train_err.append(train_summary[pct][f'{train_metric}_std'])

        if train_x:
            ax1.errorbar(train_x, train_y, yerr=train_err,
                        marker='o', linewidth=2.5, capsize=4, markersize=6,
                        label=metric_labels[metric], color=colors[i])

    ax1.set_xlabel('Sample Size (%)', fontsize=14)
    ax1.set_ylabel('Performance Score', fontsize=14)
    ax1.set_title('Training Performance\n(All Metrics)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax1.set_xlim(5, 105)
    ax1.set_ylim(0, 1)

    # Testing performance plot
    for i, metric in enumerate(base_metrics):
        test_x, test_y, test_err = [], [], []
        for pct in sorted(test_summary.keys()):
            test_metric = f'test_{metric}'
            if f'{test_metric}_mean' in test_summary[pct]:
                test_x.append(pct)
                test_y.append(test_summary[pct][f'{test_metric}_mean'])
                test_err.append(test_summary[pct][f'{test_metric}_std'])

        if test_x:
            ax2.errorbar(test_x, test_y, yerr=test_err,
                        marker='s', linewidth=2.5, capsize=4, markersize=6,
                        label=metric_labels[metric], color=colors[i], linestyle='--')

    ax2.set_xlabel('Sample Size (%)', fontsize=14)
    ax2.set_ylabel('Performance Score', fontsize=14)
    ax2.set_title('Testing Performance\n(All Metrics)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax2.set_xlim(5, 105)
    ax2.set_ylim(0, 1)

    plt.suptitle('Training vs Testing Performance - Side by Side Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    combined_plot = "/home/zhan1/EPI/train_test_combined.png"
    plt.savefig(combined_plot, dpi=300, bbox_inches='tight', facecolor='white')
    tracker.log_message(f"📈 Combined train/test plot saved to: {combined_plot}")
    plt.close()

    # 3. AUC-ROC focused comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    # Training AUC-ROC
    train_x, train_y, train_err = [], [], []
    for pct in sorted(train_summary.keys()):
        if 'train_aucroc_mean' in train_summary[pct]:
            train_x.append(pct)
            train_y.append(train_summary[pct]['train_aucroc_mean'])
            train_err.append(train_summary[pct]['train_aucroc_std'])

    # Testing AUC-ROC
    test_x, test_y, test_err = [], [], []
    for pct in sorted(test_summary.keys()):
        if 'test_aucroc_mean' in test_summary[pct]:
            test_x.append(pct)
            test_y.append(test_summary[pct]['test_aucroc_mean'])
            test_err.append(test_summary[pct]['test_aucroc_std'])

    # Plot both
    if train_x:
        ax.errorbar(train_x, train_y, yerr=train_err,
                   marker='o', markersize=10, linewidth=3, capsize=6,
                   color='steelblue', label='Training AUC-ROC',
                   markerfacecolor='white', markeredgewidth=2)

    if test_x:
        ax.errorbar(test_x, test_y, yerr=test_err,
                   marker='s', markersize=10, linewidth=3, capsize=6,
                   color='darkorange', label='Testing AUC-ROC', linestyle='--',
                   markerfacecolor='white', markeredgewidth=2)

    # Add performance gap annotations
    for i, (train_pct, test_pct) in enumerate(zip(train_x, test_x)):
        if train_pct == test_pct:
            gap = train_y[i] - test_y[i]
            ax.annotate(f'Gap: {gap:.3f}',
                       xy=(train_pct, (train_y[i] + test_y[i])/2),
                       xytext=(10, 0), textcoords='offset points',
                       fontsize=9, alpha=0.7,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax.set_xlabel('Sample Size (%)', fontsize=14)
    ax.set_ylabel('AUC-ROC Score', fontsize=14)
    ax.set_title('Training vs Testing AUC-ROC Performance\n(Performance Gap Analysis)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim(5, 105)
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    aucroc_plot = "/home/zhan1/EPI/train_test_aucroc_comparison.png"
    plt.savefig(aucroc_plot, dpi=300, bbox_inches='tight', facecolor='white')
    tracker.log_message(f"📈 AUC-ROC comparison plot saved to: {aucroc_plot}")
    plt.close()

    return [individual_plot, combined_plot, aucroc_plot]

def print_train_test_summary(train_summary, test_summary):
    """Print detailed training vs testing comparison."""
    tracker = ProgressTracker()

    tracker.log_message("\n📊 TRAINING vs TESTING PERFORMANCE COMPARISON")
    tracker.log_message("=" * 80)

    for pct in sorted(train_summary.keys()):
        if pct in test_summary:
            tracker.log_message(f"\n📈 Sample Size: {pct}%")
            tracker.log_message("-" * 50)

            # Compare key metrics
            key_metrics = ['aucroc', 'acc', 'pos_f1']
            for metric in key_metrics:
                train_key = f'train_{metric}_mean'
                test_key = f'test_{metric}_mean'

                if train_key in train_summary[pct] and test_key in test_summary[pct]:
                    train_val = train_summary[pct][train_key]
                    test_val = test_summary[pct][test_key]
                    gap = train_val - test_val
                    gap_pct = (gap / test_val) * 100

                    metric_name = metric.replace('pos_f1', 'F1').upper()
                    tracker.log_message(f"  {metric_name:8} - Train: {train_val:.4f}, Test: {test_val:.4f}, Gap: {gap:.4f} ({gap_pct:+.1f}%)")

    # Overall summary at 100%
    if 100 in train_summary and 100 in test_summary:
        tracker.log_message(f"\n🎯 FINAL PERFORMANCE (100% Sample Size)")
        tracker.log_message("-" * 50)
        tracker.log_message(f"  Training AUC-ROC: {train_summary[100]['train_aucroc_mean']:.4f}")
        tracker.log_message(f"  Testing AUC-ROC:  {test_summary[100]['test_aucroc_mean']:.4f}")
        train_test_gap = train_summary[100]['train_aucroc_mean'] - test_summary[100]['test_aucroc_mean']
        tracker.log_message(f"  Overfitting Gap:  {train_test_gap:.4f} ({(train_test_gap/test_summary[100]['test_aucroc_mean']*100):+.1f}%)")

def main():
    """Main function to create training vs testing plots."""
    tracker = ProgressTracker()
    tracker.log_message("🔬 Creating Training vs Testing Performance Analysis")

    # Load and analyze data
    train_summary, test_summary = load_and_analyze_train_test_data()

    # Create plots
    plot_files = create_train_test_comparison_plots(train_summary, test_summary)

    # Print summary
    print_train_test_summary(train_summary, test_summary)

    tracker.log_message("✨ Training vs Testing analysis complete!")
    tracker.log_message("📁 Generated plots:")
    for plot_file in plot_files:
        tracker.log_message(f"   📊 {plot_file}")

if __name__ == '__main__':
    main()