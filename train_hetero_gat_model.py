import sys
import os
import time
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append('./')

from model import HeteroGAT3, HeteroGAT4
from data_prep import heterdataPrep, connect_diagnosis_edges
from util import (
    balance_homodata_underSample,
    load_homodata,
    train_test_split_homodata,
    CustomGraphDataset,
    hetero_collate_fn,
    subsample_homodata,
)
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc, roc_auc_score
from tqdm import tqdm


def test_sample_size_with_sampling(
    file: str,
    offset: int,
    sample_fraction: float,
    batch_size,
    hidden_channels: int,
    convhidden_channels: int,
    num_layers: int,
    num_heads: int,
    lr: float,
    n_epoch: int,
    self_connect_diag: bool = False,
):
    """
    Train/evaluate with optional random subsampling of person rows.

    Args:
      file: Path to homodata H5 file.
      offset: Number of diagnosis nodes at start of `x` (e.g., 500).
      sample_fraction: Fraction of person rows to keep (0.1..1.0).
      batch_size: 'full' or int for subgraph training.
      hidden_channels, convhidden_channels, num_layers, num_heads: Model params.
      lr: Learning rate.
      n_epoch: Number of epochs.
      self_connect_diag: If True, add diagnosis-to-diagnosis edges (HeteroGAT4).
    """
    data = load_homodata(file)

    # Subsample if requested
    sample_fraction = float(sample_fraction)
    if sample_fraction < 1.0:
        data = subsample_homodata(data, offset=offset, fraction=sample_fraction)

    # Prefer Apple Metal (MPS) on macOS, then CUDA, then CPU
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    train_data, test_data = train_test_split_homodata(data, 0.8, offset)
    train_data = balance_homodata_underSample(train_data, offset)
    train_data_homo = train_data.clone()

    train_data = heterdataPrep(train_data, offset, agg=False)
    test_data = heterdataPrep(test_data, offset, agg=False)

    train_data_d2d = connect_diagnosis_edges(train_data)
    test_data_d2d = connect_diagnosis_edges(test_data)
    # test_data_balanced is not used in evaluation; skip balancing to avoid indexing edge-cases

    if self_connect_diag:
        model = HeteroGAT4(
            input_features=3,
            edge_features=1,
            hidden_channels=hidden_channels,
            convhidden_channels=convhidden_channels,
            out_channels=2,
            num_heads=num_heads,
            metadata=train_data.metadata(),
            num_layers=num_layers,
        ).to(device)
    else:
        model = HeteroGAT3(
            input_features=3,
            edge_features=1,
            hidden_channels=hidden_channels,
            convhidden_channels=convhidden_channels,
            out_channels=2,
            num_heads=num_heads,
            metadata=train_data.metadata(),
            num_layers=num_layers,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    if batch_size == 'full':
        if self_connect_diag:
            train_data = train_data_d2d
            test_data = test_data_d2d

        train_data, test_data = (
            train_data.to(device),
            test_data.to(device),
        )

        progress_bar = tqdm(range(n_epoch))
        for epoch in progress_bar:
            model.train()
            optimizer.zero_grad()
            out = model(train_data.x_dict, train_data.edge_index_dict, train_data.edge_attr_dict)
            loss = criterion(out, train_data['person'].y)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})

        print('train performance:')
        eval_mode(model, train_data)
        print('test performance:')
        eval_mode(model, test_data)
        # No balanced test evaluation

    else:
        train_dataset = CustomGraphDataset(train_data_homo, batch_size, offset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=hetero_collate_fn,
        )

        if self_connect_diag:
            train_data = train_data_d2d
            test_data = test_data_d2d

        train_data, test_data = (
            train_data.to(device),
            test_data.to(device),
        )

        for epoch in range(n_epoch):
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{n_epoch}")
            avg_loss = 0.0
            for batch in progress_bar:
                batch = batch[0]
                batch = heterdataPrep(batch, offset, agg=False)
                if self_connect_diag:
                    batch = connect_diagnosis_edges(batch)
                subgraph = batch.to(device)
                model.train()
                optimizer.zero_grad()
                out = model(subgraph.x_dict, subgraph.edge_index_dict, subgraph.edge_attr_dict)
                loss = criterion(out, subgraph['person'].y)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix({'loss': loss.item()})
                avg_loss += loss.item()

            print('avg_loss', avg_loss / max(1, len(train_dataloader)))

            if epoch % 2 == 0:
                print('train performance:')
                eval_mode(model, train_data)
                print('test performance:')
                eval_mode(model, test_data)

    return model, train_data, test_data


def eval_mode(model, data_test):
    model.eval()
    with torch.no_grad():
        out = model(data_test.x_dict, data_test.edge_index_dict, data_test.edge_attr_dict)
        _, predicted = torch.max(out, dim=1)

    y_true = data_test['person'].y.cpu()
    y_pred = predicted.cpu()
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    print("Classification Report:")
    cr_txt = classification_report(y_true, y_pred)
    print(cr_txt)
    cr = classification_report(y_true, y_pred, output_dict=True)

    probabilities = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    aucroc = roc_auc_score(y_true, probabilities)
    print(f"AUC-ROC: {aucroc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    prauc = auc(recall, precision)
    print(f"PR-AUC: {prauc:.4f}")

    return {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'aucroc': float(aucroc),
        'prauc': float(prauc),
        'pos_precision': float(cr.get('1', {}).get('precision', 0.0)),
        'pos_recall': float(cr.get('1', {}).get('recall', 0.0)),
        'pos_f1': float(cr.get('1', {}).get('f1-score', 0.0)),
        'neg_precision': float(cr.get('0', {}).get('precision', 0.0)),
        'neg_recall': float(cr.get('0', {}).get('recall', 0.0)),
        'neg_f1': float(cr.get('0', {}).get('f1-score', 0.0)),
        'support_pos': int(cr.get('1', {}).get('support', 0)),
        'support_neg': int(cr.get('0', {}).get('support', 0)),
    }

def append_metrics_csv(csv_path, args, train_metrics, test_metrics, sample_fraction, device_str, duration_sec):
    header = [
        'file','offset','sample_pct','batch_size','hidden_channels','convhidden_channels',
        'num_layers','num_heads','lr','epochs','device','duration_sec','self_connect','seed',
        'train_aucroc','train_prauc','train_acc','train_sens','train_spec','train_pos_f1',
        'test_aucroc','test_prauc','test_acc','test_sens','test_spec','test_pos_f1'
    ]
    row = [
        args.file, args.offset, int(sample_fraction*100), str(args.batch_size), args.hidden_channels,
        args.convhidden_channels, args.num_layers, args.num_heads, args.lr, args.epochs, device_str, round(float(duration_sec), 3),
        1 if args.self_connect_diag else 0, args.seed,
        train_metrics['aucroc'], train_metrics['prauc'], train_metrics['accuracy'], train_metrics['sensitivity'], train_metrics['specificity'], train_metrics['pos_f1'],
        test_metrics['aucroc'], test_metrics['prauc'], test_metrics['accuracy'], test_metrics['sensitivity'], test_metrics['specificity'], test_metrics['pos_f1']
    ]
    exists = os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if not exists:
            f.write(','.join(map(str, header)) + '\n')
        f.write(','.join(map(str, row)) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description="Run sample-size tests with adjustable sampling on offset=500 data")
    parser.add_argument('--file', type=str, default='EPI_heterodata_diag500_first.h5', help='Path to H5 file')
    parser.add_argument('--offset', type=int, default=500, help='Diagnosis offset (use 500 for diag500 file)')
    parser.add_argument('--sample_pct', type=int, default=100, help='Percent of data to sample (10-100)')
    parser.add_argument('--batch_size', default='full', help="'full' or integer subgraph size")
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','mps','cuda'], help='Device to use')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--convhidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--self_connect_diag', action='store_true', help='Use diagnosis-to-diagnosis edges (HeteroGAT4)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if isinstance(args.batch_size, str) and args.batch_size != 'full':
        try:
            args.batch_size = int(args.batch_size)
        except Exception:
            raise ValueError("batch_size must be 'full' or an integer")

    pct = max(10, min(100, int(args.sample_pct)))
    frac = pct / 100.0
    print(f"Running with sample fraction: {frac:.2f} (pct={pct})")

    # Set seeds for reproducibility
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    # Override device if requested via CLI by temporarily patching torch availability checks
    if args.device != 'auto':
        import torch as _torch
        def _pick():
            if args.device == 'mps':
                return _torch.device('mps')
            if args.device == 'cuda':
                return _torch.device('cuda')
            return _torch.device('cpu')
        _device = _pick()
        print(f"Forcing device via CLI: {_device}")
        # Monkey-patch torch.cuda.is_available / mps to steer selection inside function
        _orig_cuda = _torch.cuda.is_available
        _orig_mps = getattr(_torch.backends, 'mps').is_available if hasattr(_torch.backends, 'mps') else None
        try:
            if args.device == 'cpu':
                _torch.cuda.is_available = lambda: False
                if hasattr(_torch.backends, 'mps'):
                    _torch.backends.mps.is_available = lambda: False
            elif args.device == 'mps':
                _torch.cuda.is_available = lambda: False
                if hasattr(_torch.backends, 'mps'):
                    _torch.backends.mps.is_available = lambda: True
            elif args.device == 'cuda':
                _torch.cuda.is_available = lambda: True
                if hasattr(_torch.backends, 'mps'):
                    _torch.backends.mps.is_available = lambda: False
        except Exception:
            pass

    t0 = time.time()
    model, train_data, test_data = test_sample_size_with_sampling(
        file=args.file,
        offset=args.offset,
        sample_fraction=frac,
        batch_size=args.batch_size,
        hidden_channels=args.hidden_channels,
        convhidden_channels=args.convhidden_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        lr=args.lr,
        n_epoch=args.epochs,
        self_connect_diag=args.self_connect_diag,
    )
    # After training, capture metrics and append to CSV
    duration = time.time() - t0
    device_used = 'mps' if (hasattr(torch.backends,'mps') and torch.backends.mps.is_available()) else ('cuda' if torch.cuda.is_available() else 'cpu')
    print('Final train performance (for CSV):')
    train_metrics = eval_mode(model, train_data)
    print('Final test performance (for CSV):')
    test_metrics = eval_mode(model, test_data)
    csv_path = 'sample_size_performance.csv'
    append_metrics_csv(csv_path, args, train_metrics, test_metrics, frac, device_used, duration)
    print(f"Metrics appended to {csv_path}")

    # Restore monkey patches if any
    if args.device != 'auto':
        import torch as _torch
        try:
            # We cannot easily restore original lambdas; the process ends anyway.
            pass
        except Exception:
            pass
