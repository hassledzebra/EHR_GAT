# Claude Code Best Practices for EPI Project

## 📊 Progress Tracking and Monitoring

### Essential Tools for Every Complex Task

1. **TodoWrite Tool Usage**
   - Use for all multi-step tasks (3+ steps)
   - Update status immediately after completing each task
   - Keep only ONE task as "in_progress" at any time
   - Break complex tasks into specific, actionable items
   - Example formats:
     ```
     content: "Run model training with k-fold CV"
     activeForm: "Running model training with k-fold CV"
     ```

2. **Progress Tracker Implementation**
   - Create `progress_tracker.py` for comprehensive logging
   - Include both human-readable and machine-readable status
   - Generate timestamped progress logs
   - Provide real-time status updates with emojis for clarity
   - Enable live monitoring capabilities

3. **Background Process Management**
   - Use `run_in_background=true` for long-running tasks
   - Monitor with `BashOutput` tool regularly
   - Create timeout strategies for different task types
   - Implement process validation and health checks

## 🔬 Experimental Design Best Practices

### Statistical Rigor
- Always use k-fold cross validation (minimum 5-fold)
- Perform multiple repetitions (minimum 3) for reliability
- Include proper statistical testing (t-tests, effect sizes)
- Calculate and report confidence intervals
- Use appropriate significance levels (p < 0.05, 0.01, 0.001)

### Model Comparison Framework
```python
# Standard model comparison structure
models = {
    'baseline_models': ['Decision Tree', 'Random Forest', 'Logistic Regression', ...],
    'advanced_models': ['Custom Architecture variants'],
    'comparison_metrics': ['AUC-ROC', 'AUC-PR', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-Score']
}
```

### Parallel Computing Utilization
- Use multiprocessing for independent experiments
- Implement proper resource monitoring
- Create worker pools with `ProcessPoolExecutor`
- Monitor CPU/memory usage during execution
- Implement graceful error handling for parallel tasks

## 📁 File Organization Standards

### Directory Structure
```
project_root/
├── notebooks/          # Jupyter notebooks (.ipynb)
├── python_scripts/     # Python source files (.py)
├── data/              # Data files (.h5, .csv, etc.)
├── models/            # Trained models (.pth, .pkl)
├── results/           # Experiment outputs
├── images/            # Generated plots (.png)
├── logs/              # Log files
├── csv_files/         # CSV results
├── excel_files/       # Excel reports
└── CLAUDE.md          # This documentation
```

### Essential Output Files
Always generate these for complex analyses:
- `detailed_results.csv` - Raw experimental data
- `statistical_tests.csv` - P-values and significance tests
- `performance_summary.csv` - Mean ± std performance table
- `significance_matrix.csv` - Statistical significance matrix
- `experiment_progress.txt` - Human-readable execution log
- `experiment_status.json` - Machine-readable status

## 📈 Visualization Standards

### Comprehensive Plot Generation
1. **Individual Metrics** - Separate plots for each evaluation metric
2. **Combined Comparison** - All metrics on single plot with error bars
3. **Statistical Significance** - Heatmaps showing p-values
4. **Performance Ranking** - Sorted bar charts with confidence intervals
5. **Training vs Testing** - Overfitting analysis plots

### Plot Requirements
- Always include error bars (standard deviation)
- Use consistent color schemes across related plots
- Include sample sizes in titles/captions
- Provide clear legends and axis labels
- Save in high resolution (300 DPI minimum)
- Include statistical significance markers (*, **, ***)

## 🧪 Validation and Testing

### Pre-execution Validation
```python
def validate_experiment_setup():
    checks = [
        "Data integrity and completeness",
        "Statistical calculation functions",
        "Expected performance ranges",
        "System resource availability"
    ]
    return run_validation_tests(checks)
```

### During Execution Monitoring
- Real-time process status checking
- Resource usage monitoring (CPU, memory)
- Intermediate result validation
- Error detection and reporting
- Progress milestone tracking

### Post-execution Analysis
- Comprehensive result validation
- Statistical significance verification
- Performance trend analysis
- Comparison with expected ranges
- Final summary generation

## 📝 Documentation Standards

### Experiment Documentation
Always include:
- Clear problem statement and objectives
- Detailed methodology and parameters
- Statistical analysis approach
- Complete results with confidence intervals
- Key findings and conclusions
- Limitations and future work suggestions

### Code Documentation
- Comprehensive docstrings for all functions
- Clear variable naming conventions
- Progress logging at key checkpoints
- Error handling with informative messages
- Type hints where applicable

## 🎯 Communication and Reporting

### Progress Communication
- Use emoji-enhanced logging for clarity
- Provide percentage completion estimates
- Report intermediate findings
- Highlight significant discoveries
- Maintain professional yet engaging tone

### Final Reporting Format
```
## 🏆 EXPERIMENT SUMMARY
### Configuration: [parameters]
### Key Results: [top findings]
### Statistical Significance: [p-values]
### Conclusions: [actionable insights]
### Generated Files: [complete list]
```

## 🔄 Error Handling and Recovery

### Robust Error Management
- Implement timeout mechanisms for long-running processes
- Create fallback strategies (e.g., mock data generation)
- Provide detailed error messages with context
- Enable graceful degradation when components fail
- Log all errors with timestamps and context

### Recovery Strategies
- Checkpoint intermediate results
- Enable experiment resumption from failures
- Validate partial results before continuation
- Implement retry mechanisms with exponential backoff

## 🎨 Best Practices Summary

1. **Always use TodoWrite** for complex, multi-step tasks
2. **Implement progress tracking** with timestamped logs
3. **Use parallel computing** to maximize efficiency
4. **Apply statistical rigor** with proper validation
5. **Generate comprehensive visualizations** with error bars
6. **Create organized file structures** with clear naming
7. **Document everything** with clear explanations
8. **Validate results** at multiple checkpoints
9. **Provide engaging progress updates** with meaningful summaries
10. **Save all results** in multiple formats (CSV, JSON, PNG)

## 🚀 Future Enhancements

Consider implementing:
- Automated report generation
- Interactive visualization dashboards
- Real-time web-based progress monitoring
- Integration with version control systems
- Automated result backup and archiving
- Performance benchmarking against historical results

---

*This document should be updated as new best practices emerge from future projects.*