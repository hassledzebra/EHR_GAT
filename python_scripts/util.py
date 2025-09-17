from torch_geometric.utils import subgraph
import torch
import h5py
import math
from torch_geometric.data import HeteroData

from torch.utils.data import Dataset


def load_homodata(path):
  out = HeteroData()
  with h5py.File(path, 'r') as f:
      out.x = torch.tensor(f['x'][:])
      out.edge_index = torch.tensor(f['edge_index'][:])
      out.edge_attr = torch.tensor(f['edge_attr'][:])
      out.y = torch.tensor(f['y'][:])
  return out

def subsample_homodata(data, offset=20, fraction=1.0):
  """
  Randomly subsample a fraction of person nodes while keeping the first `offset`
  diagnosis nodes intact.

  Args:
    data: HeteroData with `x`, `edge_index`, `edge_attr`, and `y` where the first
          `offset` rows correspond to diagnosis nodes and the remaining correspond
          to person nodes.
    offset: Number of diagnosis nodes at the beginning of `x` to always retain.
    fraction: Float in (0, 1], fraction of person rows to keep.

  Returns:
    New HeteroData containing only the selected subset.
  """
  if fraction >= 1.0:
    return data.clone()

  if fraction <= 0.0:
    raise ValueError("fraction must be in (0, 1].")

  total_persons = data.y.shape[0]
  num_keep = max(1, int(math.ceil(total_persons * fraction)))

  # Randomly select `num_keep` person indices (0-based in `y` space)
  rand_perm = torch.randperm(total_persons)
  selected_person_idx = rand_perm[:num_keep]

  # Map to `x` indices by adding offset, and include the offset indices
  keep_x_idx = torch.cat([
      torch.arange(0, offset, dtype=torch.long),
      selected_person_idx + offset
  ], dim=0)

  # Build subgraph and slice tensors
  edge_index_sub, edge_attr_sub = subgraph(
      keep_x_idx.int(), data.edge_index, data.edge_attr, relabel_nodes=True)

  out = HeteroData()
  out.x = data.x[keep_x_idx].float()
  out.edge_index = edge_index_sub
  out.edge_attr = edge_attr_sub
  out.y = data.y[selected_person_idx]
  return out

def balance_homodata_underSample(data, offset = 20):
  # Count the number of samples in each class
  class_counts = torch.bincount(data.y)
  print('class counts = ',class_counts)

  # Find the minority class
  minority_class = torch.argmin(class_counts)
  majority_class = torch.argmax(class_counts)

  print('minority_class: ',minority_class)
  print('majority_class: ',majority_class)

  # Calculate the number of samples to keep from the majority class
  num_samples_to_keep = class_counts[minority_class]

  # Get indices of majority class samples
  majority_indices = (data.y == majority_class).nonzero().squeeze() + offset
#   print('majority_indices: ',majority_indices)
#   print('majority_indices_size: ',majority_indices.shape)

  # Randomly select majority class samples to keep
  random_indices = torch.randperm(majority_indices.size(0))[:num_samples_to_keep]
  selected_majority_indices = majority_indices[random_indices]

#   print('selected_majority_indices: ',selected_majority_indices)

  # Combine indices of minority and selected majority samples
  balanced_indices = torch.cat([(data.y == minority_class).nonzero().squeeze() + offset, selected_majority_indices])
  
  # Subset the data using the balanced indices
  x_balanced = data.x[balanced_indices]
  padding = torch.zeros([offset,x_balanced.shape[1]])
  x_balanced_padded = torch.cat([padding, x_balanced ], dim=0).float()
  y_balanced = data.y[balanced_indices - offset]

  extended_indices = torch.cat([torch.tensor(range(0,offset)), balanced_indices],axis=0).int() # I have to attach the diagnosis ids because the diagnosis ids are not appearing in the balanced indices, which will return empty in subgraph

  edge_index_balanced, edge_attr_balanced = subgraph(
      extended_indices,
      data.edge_index,
      data.edge_attr,
      relabel_nodes=True,
      num_nodes=int(data.x.shape[0])
  )

  # Create a new HeteroData object with balanced data
  data_balanced = HeteroData()
  data_balanced.x = x_balanced_padded
  data_balanced.edge_index = edge_index_balanced
  data_balanced.edge_attr = edge_attr_balanced
  data_balanced.y = y_balanced
  return data_balanced


def train_test_split_homodata(data, train_ratio=0.8, offset = 20):

  # Determine person count robustly
  total_persons_from_x = int(data.x.shape[0]) - int(offset)
  total_persons_from_y = int(data.y.shape[0])
  total = min(total_persons_from_x, total_persons_from_y)
  if total <= 0:
    raise ValueError("No person nodes available for split.")

  # Random permutation over [0, total)
  random_indices = torch.randperm(total)
  num_train = int(train_ratio * total)

  train_indices = random_indices[:num_train]
  test_indices = random_indices[num_train:]

  # Build subgraphs
  train_data = data.clone()
  keep_train = torch.cat([torch.arange(0, offset, dtype=torch.long), train_indices.long() + offset], dim=0)
  train_data.edge_index, train_data.edge_attr = subgraph(
     keep_train, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=int(data.x.shape[0]))
  train_data.y = data.y[train_indices]
  train_data.x = data.x[keep_train].float()
  
  test_data = data.clone()
  keep_test = torch.cat([torch.arange(0, offset, dtype=torch.long), test_indices.long() + offset], dim=0)
  test_data.edge_index, test_data.edge_attr = subgraph(
      keep_test, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=int(data.x.shape[0]))
  test_data.y = data.y[test_indices]
  test_data.x = data.x[keep_test].float()
 
  return train_data, test_data


def train_test_split_heterodata(data, train_ratio=0.8):

  # Generate a random permutation of indices
  random_indices = torch.randperm(len(data['person'].y))
  train_ratio = 0.8
  # Calculate the number of training samples
  num_train = int(train_ratio * len(data['person'].y))

  # Create the train mask based on the random permutation
  train_mask = torch.zeros(len(data['person'].y), dtype=torch.bool)
  train_mask[random_indices[:num_train]] = True

  # Create the test mask (complement of the train mask)
  test_mask = ~train_mask



  # Get indices of training and testing nodes
  train_indices = torch.where(train_mask)[0]
  test_indices = torch.where(test_mask)[0]

  train_indices = train_indices[train_indices < data['person','to','diagnosis'].edge_index[0].max()]
  test_indices = test_indices[test_indices < data['person','to','diagnosis'].edge_index[0].max()]



  # Use subgraph to extract training and testing data
  train_data = data.clone()
  train_data['person','to','diagnosis'].edge_index, train_data['person','to','diagnosis'].edge_attr = subgraph(
     train_indices, data['person','to','diagnosis'].edge_index, data['person','to','diagnosis'].edge_attr, relabel_nodes=False)
  train_data['person'].y = data['person'].y[train_indices]
  train_data['person'].x = data['person'].x[train_indices].float()
  train_data['diagnosis','to','person'].edge_index, _ = subgraph(
     train_indices, data['diagnosis','to','person'].edge_index, data['person','to','diagnosis'].edge_attr, relabel_nodes=False)

  test_data = data.clone()
  test_data['person','to','diagnosis'].edge_index, test_data['person','to','diagnosis'].edge_attr = subgraph(
      test_indices, data['person','to','diagnosis'].edge_index, data['person','to','diagnosis'].edge_attr, relabel_nodes=False)
  test_data['person'].y = data['person'].y[test_indices]
  test_data['person'].x = data['person'].x[test_indices].float()
  test_data['diagnosis','to','person'].edge_index, _ = subgraph(
     test_indices, data['diagnosis','to','person'].edge_index, data['person','to','diagnosis'].edge_attr, relabel_nodes=False)

  return train_data, test_data





class CustomGraphDataset(Dataset):
    def __init__(self, data, batch_size, offset):
        """
        Args:
            data (HeteroData): The heterogeneous graph data.
            batch_size (int): Number of nodes in each batch (excluding offset nodes).
            offset (int): Number of initial nodes to retain in every batch.
        """
        self.data = data.clone()
        self.batch_size = batch_size
        self.offset = offset
        self.num_batches = (self.data.x.shape[0] - offset) // batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        """
        Generates one batch of data containing a subgraph.

        Args:
            idx (int): Index of the batch.

        Returns:
            HeteroData: Subgraph corresponding to the given batch index.
        """
        offset_indices = torch.arange(0, self.offset)

        # Shuffle the remaining indices after the offset
        forward_indices = torch.arange(self.offset, self.data.x.shape[0])
        forward_indices = forward_indices[torch.randperm(forward_indices.size(0))]

        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size

        # Combine offset and batch indices
        batch_indices = torch.cat([
            offset_indices,
            forward_indices[start_idx:end_idx]
        ])

        # Create the subgraph using the selected indices
        subgraph_edge_index, subgraph_edge_attr = subgraph(
            batch_indices,
            self.data.edge_index,
            self.data.edge_attr,
            relabel_nodes=True,
            num_nodes=int(self.data.x.shape[0])
        )

        # Populate the subgraph data object
        subgraphed = HeteroData()
        subgraphed.x = self.data.x[batch_indices]
        subgraphed.edge_index = subgraph_edge_index
        subgraphed.edge_attr = subgraph_edge_attr
        subgraphed.y = self.data.y[forward_indices[start_idx:end_idx] - self.offset]

        return subgraphed
    

def hetero_collate_fn(batch):
    """ Custom collate function to handle HeteroData objects in a batch. """
    return batch  # Return the batch as it is since it contains HeteroData instances.
