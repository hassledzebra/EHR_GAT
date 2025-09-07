# EHR_GAT

Utilities and example models for building a heterogeneous graph attention network (GAT) on electronic health records (EHRs) using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).

## Repository structure
- `graph_prep.py`: Prepare a graph from tabular EHR data using PySpark.
- `data_prep.py`: Convert the homogeneous graph into `torch_geometric.data.HeteroData` and include helper functions such as `connect_diagnosis_edges`.
- `model.py`: Implementations of `HeteroGAT2`, `HeteroGAT3`, and `HeteroGAT4` network architectures.
- `util.py`: Data loading, balancing utilities, train/test splits, and custom dataset for batching.
- `test_sample_size.py`: Example training script for experimenting with different graph sizes.
- `extract_data.py`, `epi_gnn_extract.ipynb`: Additional utilities/notebooks for data extraction and exploration.

## Requirements

This project was developed with:

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [PySpark](https://spark.apache.org/docs/latest/api/python/) (for preprocessing)
- `h5py`, `tqdm`, `scikit-learn`

Install dependencies with:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pyspark h5py tqdm scikit-learn
```

## Data preparation

`graph_prep.py` transforms raw EHR tables into node features and edge lists. A typical workflow is:

```python
from graph_prep import graph_prep

hetero_graph = graph_prep(cohort_df, demo_df, comorbidity_df, topn=20)
```

The resulting tensors can be saved to an HDF5 file (`heterodata_diag20_first.h5`) for reuse.

`data_prep.py` provides helpers to load the HDF5 file and to convert data into a `HeteroData` object:

```python
from util import load_homodata
from data_prep import heterdataPrep

data = load_homodata("heterodata_diag20_first.h5")
hetero_data = heterdataPrep(data, offset=20, agg=False)
```

## Model training

`test_sample_size.py` demonstrates how to train the GAT models on the prepared data:

```python
from test_sample_size import test_sample_size

GAT, test_data = test_sample_size(
    file="heterodata_diag20_first.h5",
    offset=20,
    batch_size="full",
    hidden_channels=32,
    convhidden_channels=32,
    num_layers=2,
    num_heads=4,
    lr=1e-3,
    n_epoch=10,
    self_connect_diag=True,
)
```

The script prints training and evaluation metrics such as confusion matrix, sensitivity, specificity, ROC-AUC, and PR-AUC.

## Running tests

Run the unit tests (if any) with:

```bash
pytest
```

## License

This repository is distributed under the MIT License. See `LICENSE` for details if present.

