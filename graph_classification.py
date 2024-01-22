import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from multihop_gat_conv import MultiHopGATConv
from torch_geometric.nn import global_mean_pool

dataset = TUDataset(root="data/TUDataset", name="PROTEINS")

print()
print(f"Dataset: {dataset}:")
print("====================")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")

data = dataset[0]  # Get the first graph object.

print()
print(data)
print("=============================================================")

# Gather some statistics about the first graph.
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
print(f"Has isolated nodes: {data.has_isolated_nodes()}")
print(f"Has self-loops: {data.has_self_loops()}")
print(f"Is undirected: {data.is_undirected()}")

torch.manual_seed(12345)
dataset = dataset.shuffle()
total_length = len(dataset)
train_dataset = dataset[:int(total_length * 0.8)]
test_dataset = dataset[int(total_length * 0.8):]

print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f"Step {step + 1}:")
    print("=======")
    print(f"Number of graphs in the current batch: {data.num_graphs}")
    print(data)
    print()


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = MultiHopGATConv(
            dataset.num_node_features, hidden_channels, hops=1, heads=1, concat=False
        )
        self.conv2 = MultiHopGATConv(
            hidden_channels, hidden_channels, hops=3, heads=1, concat=False
        )

        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin(x)

        return x


model = GCN(hidden_channels=64)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        # Perform a single forward pass.
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        # Check against ground-truth labels.
        correct += int((pred == data.y).sum())
    # Derive ratio of correct predictions.
    return correct / len(loader.dataset)


best_test_acc = 0
for epoch in range(100):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
    print(
        f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

print(f"Best Test Acc: {best_test_acc:.4f}")
