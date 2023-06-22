import torch_geometric.nn as geom_nn
import torch.nn.functional as F
import torch.nn as nn

class GNNClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNNClassifier, self).__init__()
        self.conv1 = nn.GraphConv(num_features, 16)
        self.conv2 = nn.GraphConv(16, 32)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = geom_nn.global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)