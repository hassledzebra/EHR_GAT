#!/usr/bin/env python3
"""
Create performance plots with error bars from the experimental results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from progress_tracker import ProgressTracker

def create_performance_plots():
    """Create performance plots with error bars."""
    tracker = ProgressTracker()
    tracker.log_message("📊 Creating performance plots with matplotlib...")

    # Load summary data
    try:
        with open('/home/zhan1/EPI/performance_summary.json', 'r') as f:
            summary = json.load(f)
    except FileNotFoundError:
        tracker.log_message("❌ Summary file not found")
        return

    if not summary:
        tracker.log_message("❌ No summary data available")
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

    # Set style for better looking plots
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # Individual metric plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = ['steelblue', 'darkorange', 'green', 'red', 'purple', 'brown']

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Extract data
        x_vals = []
        y_vals = []
        err_vals = []

        for pct_str in sorted(summary.keys(), key=int):
            pct_data = summary[pct_str]
            if f'{metric}_mean' in pct_data:
                x_vals.append(int(pct_str))
                y_vals.append(pct_data[f'{metric}_mean'])
                err_vals.append(pct_data[f'{metric}_std'])

        if x_vals:
            # Create error bar plot
            ax.errorbar(x_vals, y_vals, yerr=err_vals,
                       marker='o', markersize=8, linewidth=2.5, capsize=6,
                       color=colors[i], markerfacecolor='white',
                       markeredgewidth=2, capthick=2)

            # Add trend line
            if len(x_vals) > 2:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                ax.plot(x_vals, p(x_vals), "--", alpha=0.5, color=colors[i])

            ax.set_xlabel('Sample Size (%)', fontsize=12)
            ax.set_ylabel(metric_labels[metric], fontsize=12)
            ax.set_title(f'{metric_labels[metric]} vs Sample Size\n(500 comorbidities, 800 epochs, self-connect=True, n=3 runs)',
                        fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(5, 95)
            ax.set_ylim(0, 1)

            # Add value annotations
            for x, y, err in zip(x_vals, y_vals, err_vals):
                ax.annotate(f'{y:.3f}±{err:.3f}', (x, y), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=9, alpha=0.8)

    plt.suptitle('EPI Model Performance Analysis\n500 Comorbidities, 800 Epochs, Self-Connect=True',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    individual_plot = "/home/zhan1/EPI/performance_by_metric.png"
    plt.savefig(individual_plot, dpi=300, bbox_inches='tight', facecolor='white')
    tracker.log_message(f"📈 Individual metric plots saved to: {individual_plot}")
    plt.close()

    # Combined plot
    fig, ax = plt.subplots(figsize=(14, 10))

    for i, metric in enumerate(metrics):
        # Extract data
        x_vals = []
        y_vals = []
        err_vals = []

        for pct_str in sorted(summary.keys(), key=int):
            pct_data = summary[pct_str]
            if f'{metric}_mean' in pct_data:
                x_vals.append(int(pct_str))
                y_vals.append(pct_data[f'{metric}_mean'])
                err_vals.append(pct_data[f'{metric}_std'])

        if x_vals:
            ax.errorbar(x_vals, y_vals, yerr=err_vals,
                       marker='o', linewidth=2.5, capsize=4, markersize=6,
                       label=metric_labels[metric], color=colors[i],
                       markerfacecolor='white', markeredgewidth=1.5, capthick=1.5)

    ax.set_xlabel('Sample Size (%)', fontsize=14)
    ax.set_ylabel('Performance Score', fontsize=14)
    ax.set_title('Model Performance vs Sample Size\n500 Comorbidities, 800 Epochs, Self-Connect=True (n=3 runs each)',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    ax.set_xlim(5, 95)
    ax.set_ylim(0, 1)

    # Add sample size annotations
    for x in [10, 30, 50, 70, 90]:
        ax.axvline(x, color='gray', alpha=0.2, linestyle=':')

    plt.tight_layout()
    combined_plot = "/home/zhan1/EPI/combined_performance.png"
    plt.savefig(combined_plot, dpi=300, bbox_inches='tight', facecolor='white')
    tracker.log_message(f"📈 Combined plot saved to: {combined_plot}")
    plt.close()

    # Summary statistics plot
    fig, ax = plt.subplots(figsize=(12, 8))

    sample_sizes = []
    mean_aucroc = []
    std_aucroc = []

    for pct_str in sorted(summary.keys(), key=int):
        pct_data = summary[pct_str]
        if 'test_aucroc_mean' in pct_data:
            sample_sizes.append(int(pct_str))
            mean_aucroc.append(pct_data['test_aucroc_mean'])
            std_aucroc.append(pct_data['test_aucroc_std'])

    # Create bar plot with error bars
    bars = ax.bar(sample_sizes, mean_aucroc, yerr=std_aucroc,
                  capsize=5, color='steelblue', alpha=0.7,
                  error_kw={'capthick': 2, 'capsize': 5, 'elinewidth': 2})

    # Add value labels on bars
    for i, (x, y, err) in enumerate(zip(sample_sizes, mean_aucroc, std_aucroc)):
        ax.text(x, y + err + 0.01, f'{y:.3f}±{err:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_xlabel('Sample Size (%)', fontsize=14)
    ax.set_ylabel('AUC-ROC Score', fontsize=14)
    ax.set_title('AUC-ROC Performance by Sample Size\n(Error bars represent standard deviation across 3 runs)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    aucroc_plot = "/home/zhan1/EPI/aucroc_by_sample_size.png"
    plt.savefig(aucroc_plot, dpi=300, bbox_inches='tight', facecolor='white')
    tracker.log_message(f"📈 AUC-ROC bar plot saved to: {aucroc_plot}")
    plt.close()

    tracker.log_message("✨ All plots created successfully!")

    return [individual_plot, combined_plot, aucroc_plot]

if __name__ == '__main__':
    create_performance_plots()