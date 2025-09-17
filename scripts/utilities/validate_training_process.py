#!/usr/bin/env python3
"""
Validation tests for the training process to ensure data integrity and correctness.
"""

import pandas as pd
import numpy as np
import json
from progress_tracker import ProgressTracker

def validate_existing_heterogat_results():
    """Validate the existing HeteroGAT results for consistency."""
    tracker = ProgressTracker()
    tracker.log_message("🔍 Validating existing HeteroGAT results...")

    try:
        # Load the existing HeteroGAT performance data
        df = pd.read_csv('/home/zhan1/EPI/sample_size_performance.csv')

        # Basic validation checks
        checks_passed = 0
        total_checks = 0

        # Check 1: Data completeness
        total_checks += 1
        if len(df) == 18:  # 6 sample sizes × 3 seeds
            tracker.log_message("✅ Data completeness: 18 experiments found")
            checks_passed += 1
        else:
            tracker.log_message(f"❌ Data completeness: Expected 18 experiments, found {len(df)}")

        # Check 2: Sample size distribution
        total_checks += 1
        sample_sizes = sorted(df['sample_pct'].unique())
        expected_sizes = [10, 30, 50, 70, 90, 100]
        if list(sample_sizes) == expected_sizes:
            tracker.log_message("✅ Sample sizes: All expected sizes present")
            checks_passed += 1
        else:
            tracker.log_message(f"❌ Sample sizes: Expected {expected_sizes}, found {sample_sizes}")

        # Check 3: Performance scaling trend
        total_checks += 1
        mean_aucroc_by_size = df.groupby('sample_pct')['test_aucroc'].mean()
        is_increasing = all(mean_aucroc_by_size.iloc[i] <= mean_aucroc_by_size.iloc[i+1]
                           for i in range(len(mean_aucroc_by_size)-1))
        if is_increasing:
            tracker.log_message("✅ Performance scaling: AUC-ROC increases with sample size")
            checks_passed += 1
        else:
            tracker.log_message("❌ Performance scaling: AUC-ROC does not consistently increase")

        # Check 4: Reasonable performance values
        total_checks += 1
        min_aucroc = df['test_aucroc'].min()
        max_aucroc = df['test_aucroc'].max()
        if 0.4 <= min_aucroc <= 0.6 and 0.7 <= max_aucroc <= 0.95:
            tracker.log_message(f"✅ Performance range: AUC-ROC from {min_aucroc:.3f} to {max_aucroc:.3f}")
            checks_passed += 1
        else:
            tracker.log_message(f"❌ Performance range: Unexpected AUC-ROC range {min_aucroc:.3f} to {max_aucroc:.3f}")

        # Check 5: Training vs testing gap consistency
        total_checks += 1
        df['train_test_gap'] = df['train_aucroc'] - df['test_aucroc']
        mean_gap = df['train_test_gap'].mean()
        if 0.02 <= mean_gap <= 0.05:  # Reasonable overfitting gap
            tracker.log_message(f"✅ Overfitting gap: Average {mean_gap:.3f} (reasonable)")
            checks_passed += 1
        else:
            tracker.log_message(f"❌ Overfitting gap: Average {mean_gap:.3f} (concerning)")

        # Summary
        pass_rate = checks_passed / total_checks
        tracker.log_message(f"\n📊 Validation Summary: {checks_passed}/{total_checks} checks passed ({pass_rate:.1%})")

        return checks_passed, total_checks, pass_rate

    except Exception as e:
        tracker.log_message(f"❌ Validation failed: {str(e)}")
        return 0, 1, 0.0

def test_statistical_calculations():
    """Test the statistical calculation functions."""
    tracker = ProgressTracker()
    tracker.log_message("🧮 Testing statistical calculations...")

    # Test data
    group1 = np.array([0.8, 0.82, 0.81, 0.79, 0.83])
    group2 = np.array([0.75, 0.77, 0.76, 0.74, 0.78])

    try:
        from scipy import stats

        # T-test
        t_stat, p_value = stats.ttest_ind(group1, group2)
        tracker.log_message(f"📊 T-test: t={t_stat:.3f}, p={p_value:.4f}")

        # Cohen's d
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) +
                             (len(group2) - 1) * np.var(group2, ddof=1)) /
                            (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        tracker.log_message(f"📏 Cohen's d: {cohens_d:.3f}")

        # Validate calculations
        if abs(t_stat) > 0 and 0 <= p_value <= 1 and abs(cohens_d) > 0:
            tracker.log_message("✅ Statistical calculations working correctly")
            return True
        else:
            tracker.log_message("❌ Statistical calculations have issues")
            return False

    except Exception as e:
        tracker.log_message(f"❌ Statistical test failed: {str(e)}")
        return False

def check_baseline_model_expectations():
    """Check if the baseline model results will be reasonable."""
    tracker = ProgressTracker()
    tracker.log_message("🎯 Checking baseline model expectations...")

    # Expected performance ranges for different models
    expected_ranges = {
        'Decision Tree': (0.60, 0.75),
        'Random Forest': (0.70, 0.85),
        'Logistic Regression': (0.65, 0.80),
        'Gradient Boosting': (0.72, 0.87),
        'SVM': (0.68, 0.82),
        'MLP': (0.66, 0.81)
    }

    tracker.log_message("📈 Expected AUC-ROC ranges for baseline models:")
    for model, (min_expected, max_expected) in expected_ranges.items():
        tracker.log_message(f"   {model:20s}: {min_expected:.2f} - {max_expected:.2f}")

    # Compare with HeteroGAT performance
    heterogat_aucroc = 0.834  # From our previous results
    tracker.log_message(f"\n🧠 HeteroGAT (self-connect) AUC-ROC: {heterogat_aucroc:.3f}")

    models_likely_to_beat = []
    for model, (min_expected, max_expected) in expected_ranges.items():
        if max_expected >= heterogat_aucroc:
            models_likely_to_beat.append(model)

    if models_likely_to_beat:
        tracker.log_message(f"⚔️  Models that might compete with HeteroGAT: {', '.join(models_likely_to_beat)}")
    else:
        tracker.log_message("🏆 HeteroGAT expected to outperform all baseline models")

    return expected_ranges

def monitor_system_resources():
    """Monitor system resources during training."""
    tracker = ProgressTracker()
    tracker.log_message("💻 Monitoring system resources...")

    try:
        import psutil

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        tracker.log_message(f"🔥 CPU Usage: {cpu_percent:.1f}%")

        # Memory usage
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024**3)
        memory_percent = memory.percent
        tracker.log_message(f"🧠 Memory Usage: {memory_gb:.1f}GB ({memory_percent:.1f}%)")

        # Check if resources are sufficient
        if cpu_percent > 90:
            tracker.log_message("⚠️  High CPU usage detected")
        if memory_percent > 80:
            tracker.log_message("⚠️  High memory usage detected")

        return {
            'cpu_percent': cpu_percent,
            'memory_gb': memory_gb,
            'memory_percent': memory_percent
        }

    except ImportError:
        tracker.log_message("📊 psutil not available for resource monitoring")
        return None
    except Exception as e:
        tracker.log_message(f"❌ Resource monitoring failed: {str(e)}")
        return None

def main():
    """Run all validation tests."""
    tracker = ProgressTracker()
    tracker.log_message("🔬 Starting Training Process Validation")

    # Test 1: Validate existing HeteroGAT results
    checks_passed, total_checks, pass_rate = validate_existing_heterogat_results()

    # Test 2: Test statistical calculations
    stats_working = test_statistical_calculations()

    # Test 3: Check baseline model expectations
    expected_ranges = check_baseline_model_expectations()

    # Test 4: Monitor system resources
    resources = monitor_system_resources()

    # Overall assessment
    tracker.log_message(f"\n🏁 VALIDATION SUMMARY")
    tracker.log_message("=" * 50)
    tracker.log_message(f"✅ HeteroGAT validation: {pass_rate:.1%} pass rate")
    tracker.log_message(f"✅ Statistical tests: {'Working' if stats_working else 'Failed'}")
    tracker.log_message(f"✅ Expected ranges: Defined for {len(expected_ranges)} models")
    tracker.log_message(f"✅ Resource monitoring: {'Available' if resources else 'Limited'}")

    if pass_rate >= 0.8 and stats_working:
        tracker.log_message("\n🎉 Training process validation PASSED")
        tracker.log_message("   All systems are working correctly for model comparison")
    else:
        tracker.log_message("\n⚠️  Training process validation has CONCERNS")
        tracker.log_message("   Some issues detected that may affect results quality")

if __name__ == '__main__':
    main()