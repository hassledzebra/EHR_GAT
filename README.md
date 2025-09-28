# EPI: Epilepsy Prediction using Heterogeneous Graph Attention Networks

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-Epilepsy%20Prediction-green.svg)](/)

> **Advanced machine learning framework for epilepsy prediction using heterogeneous graph attention networks with comprehensive baseline comparisons and statistical validation.**

## 🔬 Project Overview

This project implements a sophisticated machine learning pipeline for predicting epilepsy using **Heterogeneous Graph Attention Networks (HeteroGAT)**. The framework leverages patient comorbidity patterns and demographic information to predict epilepsy diagnosis, utilizing parallel computing for efficient large-scale analysis.

### Key Features

- 🧠 **HeteroGAT Architecture**: Graph-based neural networks for heterogeneous medical data
- 📊 **Comprehensive Model Comparison**: 8 models including Decision Tree, Random Forest, SVM, etc.
- 🔬 **Statistical Rigor**: K-fold cross-validation with significance testing
- ⚡ **Parallel Computing**: Optimized for multi-core processing and large datasets
- 📈 **Interactive Dashboard**: Real-time monitoring and result visualization
- 📋 **Reproducible Research**: Complete experimental framework with progress tracking


## 📊 Dataset Summary

### 🔒 Synthetic Data for Public Use
This repository includes **synthetic datasets** that maintain the same structure and statistical properties as the original clinical data while protecting patient privacy:

- **Size**: 5,000 synthetic patient records
- **Features**: 500 comorbidity conditions + demographic variables (age, gender, race)
- **Target**: Binary epilepsy diagnosis prediction
- **Class Balance**: ~22% positive cases (realistic clinical prevalence)
- **Structure**: Heterogeneous graph with patient-diagnosis relationships

### 📁 Available Datasets
- `synthetic_data/EPI_heterodata_diag500_synthetic.h5`: Full dataset (500 diagnoses)
- `synthetic_data/EPI_heterodata_diag200_synthetic.h5`: Reduced dataset (200 diagnoses)
- `synthetic_data/EPI_heterodata_diag100_synthetic.h5`: Compact dataset (100 diagnoses)
- `synthetic_data/EPI_heterodata_diag50_synthetic.h5`: Small dataset (50 diagnoses)
- `synthetic_data/EPI_heterodata_diag20_synthetic.h5`: Minimal dataset (20 diagnoses)

### 🛡️ Privacy Protection
The synthetic data is generated using advanced statistical methods that:
- Preserve realistic medical patterns and correlations
- Maintain graph structure and edge distributions
- Protect individual patient privacy (no real patient data included)
- Enable full framework testing and development

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone git@github.com:hassledzebra/EHR_GAT.git
cd EHR_GAT

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Basic Usage

#### 1. Test with Synthetic Data (Recommended First Step)
```bash
# Run comprehensive test suite
python test_with_synthetic_data.py

# Generate fresh synthetic data
python generate_synthetic_data.py
```

#### 2. Run Complete Model Comparison
```bash
# Full comparison with synthetic data (safe for testing)
python run_full_model_comparison.py

# Monitor progress in real-time
tail -f experiment_progress.txt
```

#### 3. Train HeteroGAT Model
```bash
# Train with synthetic data (default)
python train_hetero_gat_model.py \
    --file synthetic_data/EPI_heterodata_diag500_synthetic.h5 \
    --epochs 100 \
    --self-connect-diag

# For research with real data (not included in public repo)
python train_hetero_gat_model.py \
    --file data/EPI_heterodata_diag500_first.h5 \
    --epochs 800 \
    --self-connect-diag
```

#### 3. View Results Dashboard
```bash
# Start local dashboard server
python -m http.server 8000
# Open http://localhost:8000/dashboard.html
```

## 🏗️ Architecture

### HeteroGAT Model Design

```python
# Core model architecture
class HeteroGAT:
    - Person nodes: Demographic features
    - Diagnosis nodes: Comorbidity conditions
    - Edges: Patient-diagnosis relationships
    - Attention mechanism: Learn importance weights
    - Self-connect option: Diagnosis-diagnosis connections
```

### Training Pipeline

1. **Data Preprocessing**: Feature encoding, graph construction
2. **K-Fold Validation**: 5-fold CV with 3 repetitions
3. **Parallel Training**: Multi-process optimization
4. **Statistical Testing**: T-tests and effect size calculations
5. **Result Compilation**: Automated report generation

## 📁 Project Structure

```
EPI/
├── 📄 README.md                    # This file
├── 📊 dashboard.html               # Interactive results dashboard
├── 📋 requirements.txt             # Python dependencies
├── 📁 data/                        # Input datasets
│   ├── EPI_heterodata_diag500_first.h5
│   └── ...
├── 📁 scripts/
│   ├── 🔬 analysis/
│   │   ├── run_full_model_comparison.py    # Main comparison script
│   │   ├── create_model_comparison_plots.py
│   │   └── create_train_test_plots.py
│   ├── 🧠 training/
│   │   └── run_hetero_gat_training.py      # Enhanced training runner
│   └── 🛠️ utilities/
│       └── experiment_progress_tracker.py  # Progress monitoring
├── 📁 notebooks/                   # Jupyter notebooks
│   ├── baseline-testing.ipynb      # Baseline model exploration
│   └── main.ipynb                  # Primary analysis notebook
├── 📁 results/                     # Generated outputs
├── 📁 plots/                       # Visualizations
└── 📁 monitoring/                  # Progress logs and status
```

## 🔧 Configuration

### Model Parameters

```python
# Default HeteroGAT configuration
HETERO_GAT_CONFIG = {
    'hidden_channels': 128,
    'conv_hidden_channels': 64,
    'num_layers': 2,
    'num_heads': 8,
    'learning_rate': 0.001,
    'epochs': 800,
    'batch_size': 'full',
    'self_connect_diag': True
}
```

### Parallel Computing Setup

```python
# Optimize for your hardware
PARALLEL_CONFIG = {
    'n_jobs': -1,              # Use all available cores
    'backend': 'multiprocessing',
    'max_workers': 8,          # Adjust based on RAM
    'timeout': 3600           # 1 hour per experiment
}
```

## 📈 Monitoring & Progress Tracking

The framework includes comprehensive progress monitoring:

- **Real-time logs**: `experiment_progress.txt`
- **JSON status**: `experiment_status.json`
- **Performance metrics**: Live AUC/accuracy tracking
- **Resource utilization**: CPU/memory monitoring

### Example Monitoring Output
```
[2025-09-16 21:33:35] 🔬 Starting Comprehensive Model Comparison
[2025-09-16 21:33:35] 📊 Dataset: 5000 samples, 100 features
[2025-09-16 21:33:37] 🤖 Training Random Forest (Fold 1/5)
[2025-09-16 21:33:42] ✅ Random Forest completed - AUC: 0.630
```

## 🔬 Experimental Design

### Statistical Validation
- **K-Fold Cross-Validation**: 5-fold with 3 repetitions (15 experiments per model)
- **Significance Testing**: Paired t-tests between all model pairs
- **Effect Sizes**: Cohen's d for practical significance
- **Multiple Comparisons**: Bonferroni correction applied

### Sample Size Analysis
The framework supports training on different data fractions:
- 10%, 30%, 50%, 70%, 90%, 100% of available data
- Performance scaling analysis
- Learning curve generation



### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Check code style
flake8 scripts/ notebooks/
```

## 📚 Documentation

- **API Reference**: [docs/api.md](docs/api.md)
- **Methodology**: [documentation/methodologies/CLAUDE.md](documentation/methodologies/CLAUDE.md)
- **Tutorials**: [docs/tutorials/](docs/tutorials/)
- **FAQ**: [docs/faq.md](docs/faq.md)

## 🐛 Troubleshooting

### Common Issues

**Memory Errors**
```bash
# Reduce batch size or use CPU-only training
export CUDA_VISIBLE_DEVICES=""
python train_hetero_gat_model.py --batch-size 512
```

**Slow Training**
```bash
# Enable parallel processing
python run_hetero_gat_training.py --parallel --n-jobs 4
```

**Missing Dependencies**
```bash
# Install all required packages
pip install torch torch-geometric scikit-learn pandas numpy matplotlib seaborn
```


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

