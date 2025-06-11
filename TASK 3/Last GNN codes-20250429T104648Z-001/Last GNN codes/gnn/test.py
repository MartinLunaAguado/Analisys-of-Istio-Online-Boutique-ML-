import torch
from torch_geometric.datasets import Planetoid
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader


import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv, GAE
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
#from torch_geometric_temporal.signal import TemporalDataLoader, StaticGraphTemporalSignal
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Torch:", torch.__version__)
print("torch_geometric loaded:", Planetoid is not None)
print("torch_geometric_temporal loaded:", ChickenpoxDatasetLoader is not None)


# 🔄 Chargement des données
df = pd.read_csv("kiali_kpi_metrics.csv")


# 🧽 Nettoyage de time_window
df['time_window'] = df['time_window'].astype(str).str.strip()
df = df[df['time_window'] == "15S"].copy()


if df.empty:
    raise ValueError("🚨 Aucun enregistrement avec time_window == '15s'. Vérifiez le fichier CSV.")

# ✅ Nettoyage des colonnes numériques
df['error_rate'] = pd.to_numeric(df['error_rate'], errors='coerce').fillna(0.0)


# 🏷️ Attribution du statut
def assign_status(er):
    if er == 0.0:
        return  0
    elif er < 0.10:
        return 1
    else:
        return 2

df['status'] = df['error_rate'].apply(assign_status)

df.drop(df[df['istio_request_bytes'] == 0.0].index, inplace = True)
df.drop(df[df['new_request'] < 0.0].index, inplace = True)
df = df.dropna(subset=["request_rate"])

# 🔍 Préparation des données pour le modèle (uniquement anomalies)
df_anomalies = df[df['status'] != "Healthy"].copy()
df.to_csv("step1_added_label.csv", index=False)
print (df)



# Assuming your data is in a DataFrame with columns:
# ['timestamp', 'source', 'target', 'throughput', 'latency', 'error_rate', 'status']
# where status is already labeled (healthy=1, degraded=0.5, error=0)

# 1. Load and preprocess data
#df = pd.read_csv('microservice_communications.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort by timestamp
df = df.sort_values('timestamp')

# Create 1-minute time windows
df['time_window'] = df['timestamp'].dt.floor('1min')

# Exclude error_rate from features as per Option 1
features = ['throughput', 'average_latency','duration_milliseconds','request_rate']
target = 'status'

# Normalize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Get all unique microservices
all_nodes = sorted(list(set(df['source_workload'].unique()).union(set(df['destination_workload'].unique()))))
node_map = {node: i for i, node in enumerate(all_nodes)}
num_nodes = len(all_nodes)
df.head(10)



class MicroserviceCommunicationDataset:
    def __init__(self, df, node_map, features, target):
        self.df = df
        self.node_map = node_map
        self.features = features
        self.target = target
        self.time_windows = sorted(df['time_window'].unique())
        
    def __len__(self):
        return len(self.time_windows) - 1  # Predict next window
    
    def __getitem__(self, idx):
        # Use window t to predict window t+1
        current_window = self.time_windows[idx]
        next_window = self.time_windows[idx+1]
        
        # Get current window data (for node features)
        current_data = self.df[self.df['time_window'] == current_window]
        
        # Get next window data (for edge predictions)
        next_data = self.df[self.df['time_window'] == next_window]
        
        # Create edge index and features for prediction
        edge_index = []
        edge_attrs = []
        edge_labels = []
        
        for _, row in next_data.iterrows():
            src = self.node_map[row['source_workload']]
            tgt = self.node_map[row['destination_workload']]
            edge_index.append([src, tgt])
            edge_attrs.append(row[self.features].values)
            edge_labels.append(row[self.target])
        
        if not edge_index:  # Handle empty windows
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attrs = torch.empty((0, len(self.features)), dtype=torch.float)
            edge_labels = torch.empty((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
            edge_labels = torch.tensor(edge_labels, dtype=torch.long)
        
        # Create graph data object
        data = Data(
            x=torch.randn((num_nodes, 16)),  # Random initial node features
            edge_index=edge_index,
            edge_attr=edge_attrs,
            y=edge_labels
        )
        
        return data
    
    def get_edges(self, window):
        """Helper function to get edges for a specific window"""
        window_data = self.df[self.df['time_window'] == window]
        edge_index = []
        for _, row in window_data.iterrows():
            src = self.node_map[row['source_workload']]
            tgt = self.node_map[row['destination_workload']]
            edge_index.append([src, tgt])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else None

# Create dataset
dataset = MicroserviceCommunicationDataset(df, node_map, features, target)

# Split into train/val/test (70%/15%/15%)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset = [dataset[i] for i in range(train_size)]
val_dataset = [dataset[i] for i in range(train_size, train_size + val_size)]
test_dataset = [dataset[i] for i in range(train_size + val_size, len(dataset))]




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN2
from torch_geometric.nn import GCNConv

class TemporalGNN(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, num_classes,batch_size=1):
        super().__init__()
        # Temporal Graph Convolution
        self.tgnn = TGCN2(node_features, hidden_dim, batch_size=batch_size)  # Added required parameter
        
        # Edge prediction head
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        # Update node embeddings
        h = self.tgnn(data.x, data.edge_index)
        
        # Get source and target node embeddings
        src, dst = data.edge_index
        h_src = h[src]
        h_dst = h[dst]
        
        # Combine with edge features
        edge_emb = torch.cat([h_src, h_dst, data.edge_attr], dim=1)
        
        return self.edge_predictor(edge_emb)
    

# When initializing the model:
model = TemporalGNN(
    node_features=16,
    edge_features=len(features),
    hidden_dim=64,
    num_classes=3,
    batch_size=32  # Add your preferred batch size here
).to(device)







from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import GCLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    for data in train_dataset:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_dataset)

def evaluate(dataset):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            preds.append(pred.cpu())
            truths.append(data.y.cpu())
    
    preds = torch.cat(preds).numpy()
    truths = torch.cat(truths).numpy()
    
    print(classification_report(truths, preds, target_names=['healthy', 'degraded', 'error']))
    print("Macro F1-score:", f1_score(truths, preds, average='macro'))
    
    return f1_score(truths, preds, average='macro')

# Training loop
best_val_f1 = 0
for epoch in range(1, 101):
    loss = train()
    val_f1 = evaluate(val_dataset)
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pt')
    
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}')

# Load best model and test
model.load_state_dict(torch.load('best_model.pt'))
print("Final Test Results:")
test_f1 = evaluate(test_dataset)