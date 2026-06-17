"""
Graph Neural Network (GCN) for Dry Bean Classification
Builds KNN graph from features, then applies 2-layer GCN for node classification.
Pure PyTorch implementation — no PyTorch Geometric required.
"""
import numpy as np
import time, sys
sys.path.insert(0, '.')
from preprocess import preprocess_pipeline

# Load preprocessed data
print("Loading data...")
X_train, X_val, X_test, y_train, y_val, y_test, le, scaler, features = preprocess_pipeline()

# Combine train+val for graph construction (we need labels for all nodes in graph)
X_all = np.vstack([X_train, X_val])
y_all = np.concatenate([y_train, y_val])
n_train = len(y_train)
n_all = len(y_all)
n_classes = len(le.classes_)
print(f"Train nodes: {n_train}, Total graph nodes: {n_all}, Classes: {n_classes}")

# === Build KNN Graph ===
from sklearn.neighbors import NearestNeighbors

K = 10
print(f"Building KNN graph (k={K})...")
nn = NearestNeighbors(n_neighbors=K+1, metric='euclidean', n_jobs=-1)
nn.fit(X_all)
distances, indices = nn.kneighbors(X_all)

# Convert to edge index (PyTorch Geometric style: [2, num_edges])
edges_from = []
edges_to = []
for i in range(n_all):
    for j in indices[i, 1:]:  # skip self-loop (index 0)
        edges_from.append(i)
        edges_to.append(j)

edge_index = np.array([edges_from, edges_to])
print(f"Edges: {edge_index.shape[1]} ({edge_index.shape[1]/n_all:.1f} per node)")

# === Build GCN in PyTorch ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class GCNLayer(nn.Module):
    """Single GCN layer: H' = σ(D^(-1/2) A D^(-1/2) H W)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(out_dim))
    
    def forward(self, X, edge_index):
        # Message passing: aggregate neighbor features
        row, col = edge_index
        n = X.shape[0]
        
        # Normalized adjacency: A_norm = D^(-1/2) A D^(-1/2)
        # Use mean aggregation per edge direction
        deg = torch.bincount(row, minlength=n).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Sparse message passing
        messages = X[col] * norm.unsqueeze(-1)
        # Scatter sum
        aggregated = torch.zeros(n, X.shape[1], device=X.device)
        aggregated.scatter_add_(0, row.unsqueeze(-1).expand(-1, X.shape[1]), messages)
        
        # Transform
        return torch.relu(aggregated @ self.W + self.b)


class GCN(nn.Module):
    """2-layer GCN for node classification"""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNLayer(in_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, out_dim)
        self.dropout = dropout
    
    def forward(self, X, edge_index):
        h = self.conv1(X, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h  # raw logits


# === Prepare tensors ===
X_tensor = torch.FloatTensor(X_all)
y_tensor = torch.LongTensor(y_all)
edge_tensor = torch.LongTensor(edge_index)

device = torch.device('cpu')
X_tensor = X_tensor.to(device)
y_tensor = y_tensor.to(device)
edge_tensor = edge_tensor.to(device)

# Create train mask (use train portion for loss, val portion for monitoring)
train_mask = torch.zeros(n_all, dtype=torch.bool)
train_mask[:n_train] = True
val_mask = torch.zeros(n_all, dtype=torch.bool)
val_mask[n_train:] = True

# === Train ===
model = GCN(in_dim=16, hidden_dim=64, out_dim=n_classes, dropout=0.3).to(device)
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

print(f"\nTraining GCN ({sum(p.numel() for p in model.parameters()):,} params)...")
best_val_acc = 0
best_state = None

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    logits = model(X_tensor, edge_tensor)
    loss = criterion(logits[train_mask], y_tensor[train_mask])
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor, edge_tensor)
        val_pred = logits[val_mask].argmax(dim=1)
        val_acc = (val_pred == y_tensor[val_mask]).float().mean().item()
        
        train_pred = logits[train_mask].argmax(dim=1)
        train_acc = (train_pred == y_tensor[train_mask]).float().mean().item()
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# Restore best
model.load_state_dict(best_state)

# === Test ===
print("\nTesting...")
model.eval()

# For test, we need to add test nodes to the graph
# Rebuild graph with train+val+test
X_all_with_test = np.vstack([X_train, X_val, X_test])
y_all_with_test = np.concatenate([y_train, y_val, y_test])
n_test = len(y_test)

nn2 = NearestNeighbors(n_neighbors=K+1, metric='euclidean', n_jobs=-1)
nn2.fit(X_all_with_test)
dist2, idx2 = nn2.kneighbors(X_all_with_test)

edges_from2, edges_to2 = [], []
for i in range(len(X_all_with_test)):
    for j in idx2[i, 1:]:
        edges_from2.append(i)
        edges_to2.append(j)

edge_test = torch.LongTensor(np.array([edges_from2, edges_to2])).to(device)
X_test_tensor = torch.FloatTensor(X_all_with_test).to(device)

test_mask = torch.zeros(len(X_all_with_test), dtype=torch.bool)
test_mask[-n_test:] = True

with torch.no_grad():
    logits_test = model(X_test_tensor, edge_test)
    test_pred = logits_test[test_mask].argmax(dim=1)
    test_acc = (test_pred == torch.LongTensor(y_test).to(device)).float().mean().item()
    
    # Also compute train-val metrics from the original graph
    train_pred_final = logits_test[:n_train].argmax(dim=1)
    train_acc_final = (train_pred_final == torch.LongTensor(y_train).to(device)).float().mean().item()
    val_pred_final = logits_test[n_train:n_train+len(y_val)].argmax(dim=1)
    val_acc_final = (val_pred_final == torch.LongTensor(y_val).to(device)).float().mean().item()

print(f"\n{'='*60}")
print(f"  GCN Results")
print(f"{'='*60}")
print(f"  Train Acc : {train_acc_final:.4f}")
print(f"  Val Acc   : {val_acc_final:.4f}")
print(f"  Test Acc  : {test_acc:.4f}")
print(f"  Best Val  : {best_val_acc:.4f}")
print(f"  Overfit   : {train_acc_final - val_acc_final:+.4f}")
print(f"{'='*60}")
