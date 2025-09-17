import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.data import HeteroData
from torch_geometric.nn import Linear

class HeteroGAT2(torch.nn.Module):
    def __init__(self, input_features, edge_features, hidden_channels, convhidden_channels, out_channels, num_heads, metadata):
        super().__init__()

        self.gat1 = HeteroConv({
            ('person', 'to', 'diagnosis'): GATConv((-1, -1), hidden_channels, heads=num_heads, concat=True,  add_self_loops=False),
            # ('person', 'to', 'diagnosis'): GATConv((-1, -1), hidden_channels
            ('diagnosis', 'to', 'person'): GATConv((-1, -1), hidden_channels, heads=num_heads, concat=True,  add_self_loops=False),
        })
        self.gat2 = HeteroConv({
            ('person', 'to', 'diagnosis'): GATConv(hidden_channels * num_heads, convhidden_channels, heads=1, concat=False, add_self_loops=False),
            # ('person', 'to', 'diagnosis'): GATConv((-1, -1), hidden_channels
            ('diagnosis', 'to', 'person'): GATConv(hidden_channels * num_heads, convhidden_channels, heads=1, concat=False, add_self_loops=False),
        })

        self.lin1 = torch.nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
        self.lin2 = torch.nn.Linear(convhidden_channels, convhidden_channels)
        self.lin3 = torch.nn.Linear(convhidden_channels, out_channels)
        self.edge_mlp = torch.nn.Sequential(
          torch.nn.Linear(edge_features, hidden_channels),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_channels, hidden_channels)
        )
        # self.conv1 = torch.nn.Conv1d(in_channels=convhidden_channels, out_channels=out_channels, kernel_size=3, padding=1)
        # self.mlp = torch.nn.Linear(convhidden_channels, out_channels)
        # self.layer_norm = torch.nn.LayerNorm(convhidden_channels) # Add normalization layer

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
    #     print(x_dict)
        # print(edge_attr_dict)


        edge_attr_transformed = self.edge_mlp(edge_attr_dict[('person', 'to', 'diagnosis')])
        x_dict = self.gat1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {key: x + self.lin1(x) for key, x in x_dict.items()}
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        # x_dict = F.elu(x_dict)
        # x = F.dropout(x, p=0.6)
        x_dict = self.gat2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {key: x + self.lin2(x) for key, x in x_dict.items()}
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        # x_dict = F.elu(x_dict)
        # x = self.layer_norm(x) # Apply normalization
        # x = torch.permute(x, (0,2,1))
        # x = self.conv1(x)
        x = x_dict['person']
        # x = self.mlp(x)

        # print(x_dict)

        # x = F.relu(x)
        x = self.lin3(x)
        # x = F.relu(x)



        return F.log_softmax(x, dim=1)
        # return x



class HeteroGAT3(torch.nn.Module):
    def __init__(self, input_features, edge_features, hidden_channels, convhidden_channels, out_channels, num_heads, metadata, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
          self.gat1 = HeteroConv({
              ('person', 'to', 'diagnosis'): GATConv((-1, -1), hidden_channels, heads=num_heads, concat=True,  add_self_loops=False),
              # ('person', 'to', 'diagnosis'): GATConv((-1, -1), hidden_channels
              ('diagnosis', 'to', 'person'): GATConv((-1, -1), hidden_channels, heads=num_heads, concat=True,  add_self_loops=False),
          }, aggr = 'sum')

          self.convs.append(self.gat1)
        self.gat2 = HeteroConv({
            ('person', 'to', 'diagnosis'): GATConv((-1, -1), convhidden_channels, heads=1, concat=False, add_self_loops=False),
            # ('person', 'to', 'diagnosis'): GATConv((-1, -1), hidden_channels
            ('diagnosis', 'to', 'person'): GATConv((-1, -1), convhidden_channels, heads=1, concat=False, add_self_loops=False),
        }, aggr = 'sum')

        self.lin1 = torch.nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
        self.lin2 = torch.nn.Linear(convhidden_channels, convhidden_channels)
        self.lin3 = torch.nn.Linear(convhidden_channels, out_channels)
        self.edge_mlp = torch.nn.Sequential(
          torch.nn.Linear(edge_features, hidden_channels),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.edge_mlp2 = torch.nn.Sequential(
          torch.nn.Linear(edge_features, hidden_channels),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_channels, hidden_channels)
        )
        # self.conv1 = torch.nn.Conv1d(in_channels=convhidden_channels, out_channels=out_channels, kernel_size=3, padding=1)
        # self.mlp = torch.nn.Linear(convhidden_channels, out_channels)
        self.layer_norm = torch.nn.LayerNorm(convhidden_channels) # Add normalization layer

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
    #     print(x_dict)
        # print(edge_attr_dict)


        edge_attr_dict[('person', 'to', 'diagnosis')] = self.edge_mlp(edge_attr_dict[('person', 'to', 'diagnosis')])
        for conv in self.convs:
          x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
          x_dict = {key: x + self.lin1(x) for key, x in x_dict.items()}
          x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        # x_dict = F.elu(x_dict)
        # x = F.dropout(x, p=0.6)
        # edge_attr_dict[('person', 'to', 'diagnosis')] = self.edge_mlp2(edge_attr_dict[('person', 'to', 'diagnosis')])
        x_dict = self.gat2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {key: x + self.lin2(x) for key, x in x_dict.items()}
        x_dict = {key: F.elu(x)  for key, x in x_dict.items()}
        # x_dict = F.elu(x_dict)
        # x = self.layer_norm(x) # Apply normalization
        # x = torch.permute(x, (0,2,1))
        # x = self.conv1(x)
        x = x_dict['person']
        # x = self.mlp(x)

        # print(x_dict)

        # x = F.relu(x)
        # x = self.layer_norm(x) # Apply normalization

        x = self.lin3(x)
        # x = F.relu(x)



        return F.log_softmax(x, dim=1)
        # return x
    
class HeteroGAT4(torch.nn.Module):
    def __init__(self, input_features, edge_features, hidden_channels, convhidden_channels, out_channels, num_heads, metadata, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
          self.gat1 = HeteroConv({
              ('person', 'to', 'diagnosis'): GATConv((-1, -1), hidden_channels, heads=num_heads, concat=True,  add_self_loops=False),
              # ('person', 'to', 'diagnosis'): GATConv((-1, -1), hidden_channels
              ('diagnosis', 'to', 'person'): GATConv((-1, -1), hidden_channels, heads=num_heads, concat=True,  add_self_loops=False),
              ('diagnosis', 'diag_conn', 'diagnosis'): GATConv((-1, -1), hidden_channels, heads=num_heads, concat=True,  add_self_loops=False),

          }, aggr = 'sum')

          self.convs.append(self.gat1)
        self.gat2 = HeteroConv({
            ('person', 'to', 'diagnosis'): GATConv((-1, -1), convhidden_channels, heads=1, concat=False, add_self_loops=False),
            # ('person', 'to', 'diagnosis'): GATConv((-1, -1), hidden_channels
            ('diagnosis', 'to', 'person'): GATConv((-1, -1), convhidden_channels, heads=1, concat=False, add_self_loops=False),
            ('diagnosis', 'diag_conn', 'diagnosis'): GATConv((-1, -1), convhidden_channels, heads=1, concat=False,  add_self_loops=False),

        }, aggr = 'sum')

        self.lin1 = torch.nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads)
        self.lin2 = torch.nn.Linear(convhidden_channels, convhidden_channels)
        self.lin3 = torch.nn.Linear(convhidden_channels, out_channels)
        self.edge_mlp = torch.nn.Sequential(
          torch.nn.Linear(edge_features, hidden_channels),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.edge_mlp2 = torch.nn.Sequential(
          torch.nn.Linear(edge_features, hidden_channels),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_channels, hidden_channels)
        )
        # self.conv1 = torch.nn.Conv1d(in_channels=convhidden_channels, out_channels=out_channels, kernel_size=3, padding=1)
        # self.mlp = torch.nn.Linear(convhidden_channels, out_channels)
        self.layer_norm = torch.nn.LayerNorm(convhidden_channels) # Add normalization layer

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
    #     print(x_dict)
        # print(edge_attr_dict)


        edge_attr_dict[('person', 'to', 'diagnosis')] = self.edge_mlp(edge_attr_dict[('person', 'to', 'diagnosis')])
        for conv in self.convs:
          x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
          x_dict = {key: x + self.lin1(x) for key, x in x_dict.items()}
          x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        # x_dict = F.elu(x_dict)
        # x = F.dropout(x, p=0.6)
        # edge_attr_dict[('person', 'to', 'diagnosis')] = self.edge_mlp2(edge_attr_dict[('person', 'to', 'diagnosis')])
        x_dict = self.gat2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {key: x + self.lin2(x) for key, x in x_dict.items()}
        x_dict = {key: F.elu(x)  for key, x in x_dict.items()}
        # x_dict = F.elu(x_dict)
        # x = self.layer_norm(x) # Apply normalization
        # x = torch.permute(x, (0,2,1))
        # x = self.conv1(x)
        x = x_dict['person']
        # x = self.mlp(x)

        # print(x_dict)

        # x = F.relu(x)
        # x = self.layer_norm(x) # Apply normalization

        x = self.lin3(x)
        # x = F.relu(x)



        return F.log_softmax(x, dim=1)
        # return x