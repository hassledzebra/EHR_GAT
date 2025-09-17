import sys
import torch
from torch.utils.data import DataLoader
sys.path.append('./')  # Assuming data_prep.py is in the same directory
print(sys.path)
from model import HeteroGAT3, HeteroGAT4
from data_prep import heterdataPrep, connect_diagnosis_edges
from util import balance_homodata_underSample, load_homodata, train_test_split_homodata, CustomGraphDataset
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc, roc_auc_score
from tqdm import tqdm


def predict_by_window(file, pred_win, offset, batch_size, hidden_channels, convhidden_channels, num_layers, num_heads, lr, n_epoch, self_connect_diag=False):
    data = load_homodata(file)
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(data)
    data = filter_graph_len_como(data, pred_win)
    print(data)
  

    train_data, test_data = train_test_split_homodata(data, 0.8, offset)
    train_data = balance_homodata_underSample(train_data, offset)
    train_data_homo = train_data.clone()

    test_data_balanced = balance_homodata_underSample(test_data, offset)
    train_data = heterdataPrep(train_data, offset, agg=False)
    test_data = heterdataPrep(test_data, offset, agg=False)
    test_data_balanced = heterdataPrep(test_data_balanced, offset, agg=False)

    train_data_d2d = connect_diagnosis_edges(train_data)
    print(train_data_d2d)
    test_data_d2d = connect_diagnosis_edges(test_data)
    print(test_data_d2d)
    test_data_balanced_d2d = connect_diagnosis_edges(test_data_balanced)

    if self_connect_diag:
        GAT2 = HeteroGAT4(input_features=3, edge_features=1, hidden_channels=hidden_channels,
                          convhidden_channels=convhidden_channels, out_channels=2, num_heads=num_heads,
                          metadata=train_data.metadata(), num_layers=num_layers).to(device)
    else:
        GAT2 = HeteroGAT3(input_features=3, edge_features=1, hidden_channels=hidden_channels,
                          convhidden_channels=convhidden_channels, out_channels=2, num_heads=num_heads,
                          metadata=train_data.metadata(), num_layers=num_layers).to(device)

    optimizer = torch.optim.Adam(GAT2.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    if batch_size == 'full':
        if self_connect_diag:
            train_data = train_data_d2d
            test_data = test_data_d2d
            test_data_balanced = test_data_balanced_d2d

        train_data, test_data, test_data_balanced = train_data.to(device), test_data.to(device), test_data_balanced.to(device)

        progress_bar = tqdm(range(n_epoch))
        for epoch in progress_bar:
            GAT2.train()
            optimizer.zero_grad()
            out = GAT2(train_data.x_dict, train_data.edge_index_dict, train_data.edge_attr_dict)
            loss = criterion(out, train_data['person'].y)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})

        print('train performance:')
        eval_mode(GAT2, train_data)
        print('test performance:')
        eval_mode(GAT2, test_data)
        # print('test_balanced performance')
        # eval_mode(GAT2, test_data_balanced)

    else:
        # Use CustomGraphDataset for subgraph generation
        train_dataset = CustomGraphDataset(train_data_homo, batch_size, offset)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=hetero_collate_fn)


        if self_connect_diag:
            train_data = train_data_d2d
            test_data = test_data_d2d

        train_data, test_data, test_data_balanced = train_data.to(device), test_data.to(device), test_data_balanced.to(device)

        for epoch in range(n_epoch):
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{n_epoch}")
            avg_loss = 0.0
            for batch in progress_bar:
                batch = batch[0]
                batch = heterdataPrep(batch, offset, agg=False)
                if self_connect_diag:
                  batch = connect_diagnosis_edges(batch)
                # print(batch)
                subgraph = batch.to(device)  # Extract and send the subgraph to the device
                GAT2.train()
                optimizer.zero_grad()
                out = GAT2(subgraph.x_dict, subgraph.edge_index_dict, subgraph.edge_attr_dict)
                loss = criterion(out, subgraph['person'].y)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix({'loss': loss.item()})
                avg_loss = avg_loss + loss.item()

            print('avg_loss',avg_loss/len(train_dataloader))

            if epoch % 2 == 0: # print evaluation report every 2 epochs
                print('train performance:')
                eval_mode(GAT2, train_data)
                print('test performance:')
                eval_mode(GAT2, test_data)
                # print('test_balanced performance')
                # eval_mode(GAT2, test_data_balanced)

    return GAT2

def eval_mode(GAT2, data_test):
    # Evaluate the model
    GAT2.eval()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    # data_test = data_test.to(device)
    with torch.no_grad():
        out = GAT2(data_test.x_dict, data_test.edge_index_dict, data_test.edge_attr_dict)
        _, predicted = torch.max(out, dim=1)  # Predictions for persons only

    # Get the confusion matrix
    cm = confusion_matrix(data_test['person'].y.cpu(), predicted.cpu())
    print("Confusion Matrix:")
    print(cm)

    # Extract True Positives, False Positives, False Negatives, and True Negatives
    tn, fp, fn, tp = cm.ravel()

    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(data_test['person'].y.cpu(), predicted.cpu()))

    # Extract probabilities for the positive class (class 1)
    probabilities = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

    # Calculate the AUC-ROC
    aucroc = roc_auc_score(data_test['person'].y.cpu(), probabilities)
    print(f"AUC-ROC: {aucroc:.4f}")

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(data_test['person'].y.cpu(), probabilities)

    # Calculate PRAUC
    prauc = auc(recall, precision)
    print(f"PR-AUC: {prauc:.4f}")


def filter_graph_len_como(homodata, pred_win):
   filtered = homodata.clone()
   print(homodata.edge_attr[:,0])
   mask = homodata.edge_attr[:,0] >= pred_win  # Create a mask for edges that meet the threshold condition
   print(mask)
   print(homodata.edge_index)
   filtered.edge_attr = homodata.edge_attr[mask]  # Apply mask to edge attributes
   filtered.edge_index = homodata.edge_index[:, mask]  # Apply mask to edge index
   
   return filtered


