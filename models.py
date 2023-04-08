import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, hidden_layers=2, dp_rate=0, **kwargs):
        super().__init__()
        layers = []
        in_channels, out_channels = d_in, d_hidden
        for l_idx in range(hidden_layers):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)]
            in_channels = d_hidden
        layers += [nn.Linear(d_hidden, d_out), nn.Sigmoid()]
        self.layers = nn.Sequential()
        for m in layers:
            self.layers.append(m)
    
    def forward(self, x):
        return self.layers(x)

        
class GNNModel(nn.Module):
    #dummy model not using edge information
    #d_glob is the dimension of one-hot-encoded reactants
    def __init__(self, d_in, d_hidden, d_out, d_glob, num_layers=2, dp_rate=0, **kwargs):
        super().__init__()
        conv_layer = geom_nn.GCNConv
        
        graph_layers = []
        in_channels, out_channels = d_in, d_hidden
        for l_idx in range(num_layers-1):
            graph_layers += [
                conv_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)]
            in_channels = d_hidden
        graph_layers += [conv_layer(in_channels=in_channels, out_channels=d_out,**kwargs)]
        self.graph_layers = nn.ModuleList(graph_layers)
        
        self.head = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(d_out + d_glob, 10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid()
        )

    #glob_vecs are the one hot encoding of reactants
    def forward(self, x, edge_index, edge_attr, batch_idx, glob_vecs):
        for l in self.graph_layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index.long())
            else:
                x = l(x)
        x = geom_nn.global_mean_pool(x, batch_idx)
        x = torch.cat((x, glob_vecs.reshape(-1,10)),1)
        x = self.head(x)
        return x

class GNNModel_sg(nn.Module):
    #dummy model not using edge information
    #for data with ligand and reactants put in the same graph
    def __init__(self, d_in, d_hidden, d_out, num_layers=2, dp_rate=0, **kwargs):
        super().__init__()
        conv_layer = geom_nn.GCNConv
        
        graph_layers = []
        in_channels, out_channels = d_in, d_hidden
        for l_idx in range(num_layers-1):
            graph_layers += [
                conv_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)]
            in_channels = d_hidden
        graph_layers += [conv_layer(in_channels=in_channels, out_channels=d_out,**kwargs)]
        self.graph_layers = nn.ModuleList(graph_layers)
        
        self.head = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(d_out, 10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr, batch_idx):
        for l in self.graph_layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index.long())
            else:
                x = l(x)
        x = geom_nn.global_mean_pool(x, batch_idx)
        x = self.head(x)
        return x

class GNNModel_sg_edge_attr(nn.Module):
    #uses bond features as well
    #for data with ligand and reactants put in the same graph
    def __init__(self, d_in, d_hidden, d_out, edge_dim, num_layers=2, dp_rate=0, **kwargs):
        super().__init__()
        graph_layers = []
        in_channels, out_channels = d_in, d_hidden
        for l_idx in range(num_layers-1):
            graph_layers += [
                geom_nn.GINEConv(nn.Sequential(nn.Linear(in_channels, out_channels)), train_eps = True, edge_dim = edge_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)]
            in_channels = d_hidden
        graph_layers += [geom_nn.GINEConv(nn.Sequential(nn.Linear(in_channels, d_out)), train_eps = True, edge_dim = edge_dim)]
        self.graph_layers = nn.ModuleList(graph_layers)
        
        self.head = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(d_out, 10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid()
        )

    #glob_vecs are the one hot encoding of reactants
    def forward(self, x, edge_index, edge_attr, batch_idx):
        for l in self.graph_layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index.long(),edge_attr = edge_attr)
            else:
                x = l(x)
        x = geom_nn.global_mean_pool(x, batch_idx)
        x = self.head(x)
        return x

class GNNModel_sg_edge_attr_ddG(nn.Module):
    #uses bond features as well
    #for data with ligand and reactants put in the same graph
    def __init__(self, d_in, d_hidden, d_out, edge_dim, num_layers=2, dp_rate=0, **kwargs):
        super().__init__()
        graph_layers = []
        in_channels, out_channels = d_in, d_hidden
        for l_idx in range(num_layers-1):
            graph_layers += [
                geom_nn.GINEConv(nn.Sequential(nn.Linear(in_channels, out_channels)), train_eps = True, edge_dim = edge_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)]
            in_channels = d_hidden
        graph_layers += [geom_nn.GINEConv(nn.Sequential(nn.Linear(in_channels, d_out)), train_eps = True, edge_dim = edge_dim)]
        self.graph_layers = nn.ModuleList(graph_layers)
        
        self.head = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(d_out, 10),
            nn.ReLU(),
            nn.Linear(10,1),
        )

    #glob_vecs are the one hot encoding of reactants
    def forward(self, x, edge_index, edge_attr, batch_idx):
        for l in self.graph_layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index.long(),edge_attr = edge_attr)
            else:
                x = l(x)
        x = geom_nn.global_mean_pool(x, batch_idx)
        x = self.head(x)
        return x