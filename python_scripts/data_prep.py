import h5py
from torch_geometric.data import HeteroData
import torch

def heterdataPrep(data, offset = 20, agg = False): 

  x = data.x
  edge_index = data.edge_index
  edge_attr = data.edge_attr
  y = data.y

  edge_index[0] -= offset

  out = HeteroData()
  out['diagnosis'].x =torch.tensor(range(0,offset), dtype=torch.float).unsqueeze(1) # the index of diagnosis matters. It is a feature of the diagnosis node
  out['person'].x = x[:,1:][offset:].float()

  if agg:
    out['diagnosis','to','person'].edge_index, out['person','to','diagnosis'].edge_attr = agg_edge(edge_index, edge_attr)
    out['person','to','diagnosis'].edge_index = out['diagnosis','to','person'].edge_index.clone()
    out['person','to','diagnosis'].edge_index[0] = out['diagnosis','to','person'].edge_index[1]
    out['person','to','diagnosis'].edge_index[1] = out['diagnosis','to','person'].edge_index[0]
  else:
    out['person','to','diagnosis'].edge_index, out['person','to','diagnosis'].edge_attr = edge_index, edge_attr
    out['diagnosis','to','person'].edge_index = out['person','to','diagnosis'].edge_index.clone()
    out['diagnosis','to','person'].edge_index[0] = out['person','to','diagnosis'].edge_index[1]
    out['diagnosis','to','person'].edge_index[1] = out['person','to','diagnosis'].edge_index[0]
  out['person'].y = y

  return out

def agg_edge(edge_index, edge_attr):
  # aggregate the edge according to max edge attr
  unique_edges = []
  for i in range(edge_index.shape[1]):
    u, v = edge_index[0, i].item(), edge_index[1, i].item()
    attr = edge_attr[i].item()
    unique_edges.append(((min(u, v), max(u, v)), attr))

    # Aggregate edges based on the highest edge attribute
  aggregated_edges = {}
  for (u, v), attr in unique_edges:
    if (u, v) not in aggregated_edges or attr > aggregated_edges[(u, v)]:
      aggregated_edges[(u, v)] = attr


    # Create a new edge index and edge attribute tensor
  new_edge_index = torch.tensor(list(aggregated_edges.keys())).t().contiguous()
  new_edge_attr = torch.tensor(list(aggregated_edges.values()), dtype=torch.float).unsqueeze(1)

  return new_edge_index, new_edge_attr
  
def connect_diagnosis_edges(data):
    """
    Connects diagnosis nodes to form edges between them.

    Args:
        data: A PyTorch Geometric HeteroData object.  It's assumed
              that the data has a 'diagnosis' node type and edge types
              that relate to diagnosis nodes.
              The function will create new edges between diagnosis nodes.

    Returns:
        A new HeteroData object with added diagnosis-to-diagnosis edges.
    """

    if not isinstance(data, HeteroData):
        raise TypeError("Input data must be a HeteroData object.")

    new_data = data.clone()

    # Get the number of diagnosis nodes
    num_diagnoses = new_data['diagnosis'].num_nodes
    # Use range to represent diagnosis node ids:
    diagnosis_nodes = torch.arange(num_diagnoses) # This line is changed to use range as node ids.

    # Create edges between all pairs of diagnosis nodes
    source_nodes = []
    target_nodes = []
    for i in range(num_diagnoses):
        for j in range(i + 1, num_diagnoses):  #Avoid self-loops and duplicate edges
            source_nodes.append(diagnosis_nodes[i])
            target_nodes.append(diagnosis_nodes[j])

    if 'diagnosis__to__diagnosis' not in new_data.edge_types: #Add edges if not existing
        new_data['diagnosis', 'diag_conn', 'diagnosis'].edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    else: #concatenate edges if edges exist
      existing_edge_index = new_data['diagnosis', 'diag_conn', 'diagnosis'].edge_index
      new_edge_index = torch.cat([existing_edge_index, torch.tensor([source_nodes, target_nodes], dtype=torch.long)], dim=1)
      new_data['diagnosis', 'diag_conn', 'diagnosis'].edge_index = new_edge_index

    return new_data
