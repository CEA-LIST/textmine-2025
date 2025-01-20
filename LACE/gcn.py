import torch.nn as nn
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 activation,
                 residual):
        super(GCN, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation

        # Première couche GCN
        self.gcn_layers.append(GraphConv(in_dim, num_hidden, activation=self.activation))

        # Couches intermédiaires GCN
        for _ in range(1, num_layers - 1):
            self.gcn_layers.append(GraphConv(num_hidden, num_hidden, activation=self.activation))

        # Dernière couche GCN (sans activation)
        self.gcn_layers.append(GraphConv(num_hidden, num_classes, activation=None))

        # Optionnel : ajouter une connexion résiduelle
        self.residual = residual

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h_in = h  # Pour la connexion résiduelle
            h = self.gcn_layers[l](self.g, h)
            if self.residual and h_in.shape == h.shape:
                h = h + h_in
        return h
