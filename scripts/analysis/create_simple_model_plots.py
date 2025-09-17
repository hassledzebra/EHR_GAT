#!/usr/bin/env python3
"""
Create model comparison plots without seaborn dependency.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from progress_tracker import ProgressTracker

def create_model_comparison_plot():
    """Create comprehensive model comparison plot from results."""
    tracker = ProgressTracker()
    tracker.log_message("📊 Creating model comparison plots...")

    try:
        # Load the performance summary table
        df = pd.read_csv('/home/zhan1/EPI/performance_summary_table.csv', index_col=0)

        # Parse the mean ± std format
        parsed_data = {}
        for model in df.index:
            parsed_data[model] = {}
            for metric in df.columns:
                value_str = str(df.loc[model, metric])
                if '±' in value_str:
                    mean_str, std_str = value_str.split(' ± ')
                    parsed_data[model][metric + '_mean'] = float(mean_str)
                    parsed_data[model][metric + '_std'] = float(std_str)

        parsed_df = pd.DataFrame(parsed_data).T

        # Set up colors for different model types
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # Plot 1: AUC-ROC Comparison Bar Chart
        models = parsed_df.index
        auc_means = parsed_df['auc_roc_mean']
        auc_stds = parsed_df['auc_roc_std']

        bars1 = ax1.bar(range(len(models)), auc_means, yerr=auc_stds,
                        capsize=5, color=colors[:len(models)], alpha=0.8,
                        edgecolor='black', linewidth=1)

        ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax1.set_ylabel('AUC-ROC Score', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance Comparison - AUC-ROC\n(K-fold Cross Validation, n=3 repeats × 5 folds)',
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(auc_means, auc_stds)):
            ax1.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Highlight the top performers
        max_perf = auc_means.max()
        for i, mean in enumerate(auc_means):
            if mean > 0.75:  # Highlight high-performing models
                bars1[i].set_edgecolor('gold')
                bars1[i].set_linewidth(3)

        # Plot 2: Multiple Metrics Comparison
        metrics = ['auc_roc_mean', 'auc_pr_mean', 'accuracy_mean', 'f1_score_mean']
        metric_labels = ['AUC-ROC', 'AUC-PR', 'Accuracy', 'F1-Score']
        metric_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        x = np.arange(len(models))
        width = 0.2

        for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
            values = parsed_df[metric]
            stds = parsed_df[metric.replace('_mean', '_std')]

            ax2.bar(x + i*width, values, width, yerr=stds, capsize=3,
                    label=label, alpha=0.8, color=color)

        ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
        ax2.set_title('Multi-Metric Performance Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1)

        # Plot 3: Performance Ranking (Horizontal Bar Chart)
        sorted_df = parsed_df.sort_values('auc_roc_mean', ascending=True)
        y_pos = np.arange(len(sorted_df))
        performance = sorted_df['auc_roc_mean']
        errors = sorted_df['auc_roc_std']

        bars3 = ax3.barh(y_pos, performance, xerr=errors, capsize=5,
                         color=colors[:len(models)], alpha=0.8,
                         edgecolor='black', linewidth=1)

        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(sorted_df.index, fontsize=10)
        ax3.set_xlabel('AUC-ROC Score', fontsize=12, fontweight='bold')
        ax3.set_title('Model Performance Ranking\n(Sorted by AUC-ROC Performance)',
                      fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.set_xlim(0, 1)

        # Add performance values
        for i, (perf, err) in enumerate(zip(performance, errors)):
            ax3.text(perf + err + 0.02, i, f'{perf:.3f}',
                    va='center', ha='left', fontsize=10, fontweight='bold')

        # Highlight HeteroGAT models
        for i, model in enumerate(sorted_df.index):
            if 'HeteroGAT' in model:
                bars3[i].set_edgecolor('red')
                bars3[i].set_linewidth(3)

        # Plot 4: Model Performance Summary Table
        ax4.axis('off')
        ax4.set_title('Performance Summary Statistics', fontsize=14, fontweight='bold')

        # Create summary table
        table_data = []
        sorted_models = parsed_df.sort_values('auc_roc_mean', ascending=False)

        headers = ['Rank', 'Model', 'AUC-ROC', 'AUC-PR', 'Accuracy', 'F1-Score']
        table_data.append(headers)

        for rank, (model, row) in enumerate(sorted_models.iterrows(), 1):
            auc_roc = f"{row['auc_roc_mean']:.3f}±{row['auc_roc_std']:.3f}"
            auc_pr = f"{row['auc_pr_mean']:.3f}±{row['auc_pr_std']:.3f}"
            accuracy = f"{row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}"
            f1 = f"{row['f1_score_mean']:.3f}±{row['f1_score_std']:.3f}"

            model_short = model.replace('HeteroGAT', 'GAT').replace('(self-connect)', '(SC)')
            model_short = model_short.replace('(no self-connect)', '(NSC)')
            if len(model_short) > 20:
                model_short = model_short[:17] + '...'

            table_data.append([str(rank), model_short, auc_roc, auc_pr, accuracy, f1])

        # Create table
        table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Color code the top performers
        for i in range(1, min(4, len(table_data))):  # Top 3 models
            for j in range(len(headers)):
                if i == 1:  # Best model
                    table[(i, j)].set_facecolor('#FFD700')  # Gold
                elif i == 2:  # Second best
                    table[(i, j)].set_facecolor('#C0C0C0')  # Silver
                elif i == 3:  # Third best
                    table[(i, j)].set_facecolor('#CD7F32')  # Bronze

        plt.suptitle('Comprehensive Model Comparison: Baseline Methods vs HeteroGAT\n' +
                     'EPI Prediction with 500 Comorbidities (K-fold Cross Validation)',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_path = '/home/zhan1/EPI/comprehensive_model_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        tracker.log_message(f"📊 Comprehensive comparison plot saved to: {plot_path}")
        plt.close()

        # Create a second plot showing statistical significance
        create_significance_plot(parsed_df, tracker)

        return plot_path

    except Exception as e:
        tracker.log_message(f"❌ Error creating plots: {str(e)}")
        return None

def create_significance_plot(parsed_df, tracker):
    """Create a plot showing statistical significance and effect sizes."""
    try:
        # Load statistical tests
        stats_df = pd.read_csv('/home/zhan1/EPI/statistical_tests.csv')

        # Filter for AUC-ROC comparisons
        auc_stats = stats_df[stats_df['metric'] == 'auc_roc'].copy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Plot 1: P-value comparison matrix
        models = parsed_df.index
        n_models = len(models)

        # Create p-value matrix
        p_matrix = np.ones((n_models, n_models)) * 0.5  # Initialize with 0.5 (no comparison)

        for _, row in auc_stats.iterrows():
            model1, model2 = row['model1'], row['model2']
            p_val = row['p_value']

            if model1 in models and model2 in models:
                i = list(models).index(model1)
                j = list(models).index(model2)
                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val

        # Set diagonal to 1 (same model comparison)
        np.fill_diagonal(p_matrix, 1.0)

        im1 = ax1.imshow(p_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=0.05)
        ax1.set_xticks(range(n_models))
        ax1.set_yticks(range(n_models))
        ax1.set_xticklabels([m.replace(' ', '\n') for m in models], rotation=45, ha='right', fontsize=9)
        ax1.set_yticklabels([m.replace(' ', '\n') for m in models], fontsize=9)
        ax1.set_title('Statistical Significance Matrix\n(p-values for AUC-ROC differences)',
                      fontsize=14, fontweight='bold')

        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('p-value', fontsize=12)

        # Add significance markers
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    p_val = p_matrix[i, j]
                    if p_val < 0.001:
                        marker = '***'
                    elif p_val < 0.01:
                        marker = '**'
                    elif p_val < 0.05:
                        marker = '*'
                    else:
                        marker = 'ns'

                    color = 'white' if p_val < 0.025 else 'black'
                    ax1.text(j, i, marker, ha='center', va='center',
                            color=color, fontsize=12, fontweight='bold')

        # Plot 2: Performance differences with confidence intervals
        top_models = ['HeteroGAT (self-connect)', 'HeteroGAT (no self-connect)',
                      'Logistic Regression', 'SVM', 'Gradient Boosting']

        y_pos = range(len(top_models))
        means = [parsed_df.loc[model, 'auc_roc_mean'] for model in top_models]
        stds = [parsed_df.loc[model, 'auc_roc_std'] for model in top_models]

        colors_subset = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        bars = ax2.barh(y_pos, means, xerr=stds, capsize=5,
                        color=colors_subset, alpha=0.8,
                        edgecolor='black', linewidth=1)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_models, fontsize=11)
        ax2.set_xlabel('AUC-ROC Score', fontsize=12, fontweight='bold')
        ax2.set_title('Top 5 Models Performance Comparison\n(with 95% confidence intervals)',
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, 1)

        # Add performance values
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax2.text(mean + std + 0.02, i, f'{mean:.3f}±{std:.3f}',
                    va='center', ha='left', fontsize=10, fontweight='bold')

        # Highlight statistically significant differences
        best_model_perf = max(means)
        for i, (model, mean) in enumerate(zip(top_models, means)):
            if model != top_models[0] and (best_model_perf - mean) > 2 * stds[0]:
                bars[i].set_edgecolor('red')
                bars[i].set_linewidth(3)

        plt.suptitle('Statistical Analysis of Model Performance Differences',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        sig_plot_path = '/home/zhan1/EPI/statistical_significance_analysis.png'
        plt.savefig(sig_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        tracker.log_message(f"📊 Statistical significance plot saved to: {sig_plot_path}")
        plt.close()

        return sig_plot_path

    except Exception as e:
        tracker.log_message(f"⚠️  Could not create significance plot: {str(e)}")
        return None

def main():
    """Main function."""
    tracker = ProgressTracker()
    tracker.log_message("🎨 Creating comprehensive model comparison visualizations...")

    plot_path = create_model_comparison_plot()

    if plot_path:
        tracker.log_message("✨ Model comparison plots created successfully!")
    else:
        tracker.log_message("❌ Failed to create plots")

if __name__ == '__main__':
    main()