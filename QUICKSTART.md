# 🚀 EPI Quick Start Guide

This guide will get you up and running with the EPI epilepsy prediction framework in under 10 minutes.

## 📋 Prerequisites

- Python 3.7+ installed
- 8GB+ RAM recommended
- 4+ CPU cores for parallel processing

## ⚡ Installation

### Option 1: Quick Setup
```bash
# Clone and install
git clone https://github.com/your-org/epi-prediction.git
cd epi-prediction
pip install -r requirements.txt
```

### Option 2: Development Setup
```bash
# Install in editable mode with development tools
pip install -e .[dev]
```

## 🎯 Quick Examples

### 1. Run Complete Model Comparison (Recommended First Step)

```bash
# Start the full comparison with progress monitoring
python run_full_model_comparison.py

# In another terminal, monitor progress
tail -f experiment_progress.txt
```

**Expected Output:**
```
🔬 Starting Comprehensive Model Comparison with K-Fold CV
📊 Dataset: 5000 samples, 100 features
🤖 Training Random Forest (Fold 1/5)
✅ Random Forest completed - AUC: 0.630
...
🏆 HeteroGAT (self-connect) | AUC-ROC: 0.8283 ± 0.0114
```

**Runtime:** ~7 minutes on 8-core machine

### 2. Train Individual HeteroGAT Model

```bash
# Quick training with default parameters
python train_hetero_gat_model.py \
    --file data/EPI_heterodata_diag500_first.h5 \
    --offset 500 \
    --sample-fraction 0.5 \
    --epochs 100 \
    --self-connect-diag

# Full training (takes longer but better results)
python train_hetero_gat_model.py \
    --file data/EPI_heterodata_diag500_first.h5 \
    --offset 500 \
    --sample-fraction 1.0 \
    --epochs 800 \
    --self-connect-diag
```

### 3. View Interactive Dashboard

```bash
# Option A: Simple HTTP server
python -m http.server 8000
# Open http://localhost:8000/dashboard.html

# Option B: Direct file opening
# Open dashboard.html in your browser
```

## 📊 Understanding the Results

### Performance Metrics Explained

- **AUC-ROC**: Area under ROC curve (higher = better, max = 1.0)
- **AUC-PR**: Area under Precision-Recall curve (good for imbalanced data)
- **Accuracy**: Overall correct predictions
- **F1-Score**: Harmonic mean of precision and recall

### What Good Results Look Like

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| AUC-ROC | >0.80 | 0.70-0.80 | 0.60-0.70 | <0.60 |
| AUC-PR | >0.70 | 0.50-0.70 | 0.30-0.50 | <0.30 |

**Our Results:** HeteroGAT achieves **0.828 AUC-ROC** - Excellent performance! 🎉

## 🔧 Common Configurations

### For Different Hardware

**High-end machine (16+ cores, 32GB+ RAM):**
```bash
python run_full_model_comparison.py --n-jobs 16 --batch-size full
```

**Laptop/modest hardware:**
```bash
python run_full_model_comparison.py --n-jobs 4 --batch-size 1024
```

**CPU-only (no GPU):**
```bash
export CUDA_VISIBLE_DEVICES=""
python train_hetero_gat_model.py --device cpu
```

### For Different Sample Sizes

**Quick test (10% data):**
```bash
python train_hetero_gat_model.py --sample-fraction 0.1 --epochs 100
```

**Full analysis (100% data):**
```bash
python train_hetero_gat_model.py --sample-fraction 1.0 --epochs 800
```

## 📈 Monitoring Your Experiments

### Real-time Progress
```bash
# Watch the main progress log
tail -f experiment_progress.txt

# Monitor resource usage
top -p $(pgrep -f python)
```

### Check Status Programmatically
```python
import json
with open('experiment_status.json') as f:
    status = json.load(f)
print(f"Progress: {status['completed']}/{status['total']} ({status['percentage']:.1f}%)")
```

## 🎨 Customizing the Analysis

### Custom Model Parameters
```python
# Edit train_hetero_gat_model.py
CUSTOM_CONFIG = {
    'hidden_channels': 256,      # Increase for more capacity
    'num_heads': 16,            # More attention heads
    'learning_rate': 0.0005,    # Lower for stability
    'epochs': 1000              # More training
}
```

### Custom Sample Sizes
```bash
# Test multiple sample sizes
for size in 0.1 0.3 0.5 0.7 1.0; do
    python train_hetero_gat_model.py --sample-fraction $size --output-suffix "_size_$size"
done
```

## 🔍 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'torch'"**
```bash
pip install torch torch-geometric
```

**"CUDA out of memory"**
```bash
export CUDA_VISIBLE_DEVICES=""  # Use CPU only
# or reduce batch size
python train_hetero_gat_model.py --batch-size 512
```

**"Permission denied" errors**
```bash
chmod +x *.py
```

**Slow performance**
```bash
# Check if running in parallel
python run_full_model_comparison.py --n-jobs 8  # Use more cores
```

### Getting Help

1. **Check the logs:** Look in `experiment_progress.txt` for detailed error messages
2. **Reduce complexity:** Try smaller sample sizes first
3. **Hardware check:** Ensure you have enough RAM (8GB minimum)
4. **Python version:** Verify Python 3.7+ with `python --version`

## 🎯 Next Steps

Once you've successfully run the basic examples:

1. **Explore the dashboard** - Understand your results visually
2. **Read the full README** - Learn about the methodology
3. **Try different parameters** - Optimize for your specific use case
4. **Examine the notebooks** - See detailed analysis examples
5. **Contribute** - Add your own models or improvements

## 💡 Pro Tips

- **Start small:** Always test with 10% data first
- **Monitor resources:** Check CPU/memory usage during training
- **Save progress:** The framework automatically saves checkpoints
- **Compare models:** The comparison script gives you the best overview
- **Use parallel processing:** It's 6x faster than single-threaded

---

**Need more help?** Check out the [full documentation](README.md) or open an issue on GitHub!