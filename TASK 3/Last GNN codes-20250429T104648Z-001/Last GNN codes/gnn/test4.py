import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, f1_score  # Added f1_score import
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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
    if er < 0.01:
        return 0
    elif er < 0.1:
        return 1
    else:
        return 2

def assign_status(er):
    if er < 0.15:
        return 0
    
    else:
        return 1



df['status'] = df['error_rate'].apply(assign_status)
df.drop(df[df['istio_request_bytes'] == 0.0].index, inplace = True)
df.drop(df[df['new_request'] < 0.0].index, inplace = True)
df = df.dropna(subset=["request_rate"])
df.to_csv("data2.csv", index=False)







class MicroserviceDataset:
    def __init__(self, df, node_map, features, target):
        self.df = df
        self.node_map = node_map
        self.features = features
        self.target = target
        
    def __len__(self):
        return 1  # We'll treat the whole dataset as one graph
    
    def __getitem__(self, idx):
        edge_index = []
        edge_attr = []
        edge_labels = []
        
        # Create edges with features and labels
        for _, row in self.df.iterrows():
            src = self.node_map[row['source_workload']]
            tgt = self.node_map[row['destination_workload']]
            edge_index.append([src, tgt])
            edge_attr.append(row[self.features].values)
            edge_labels.append(row[self.target])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_labels = torch.tensor(edge_labels, dtype=torch.long)
        
        # Create node features (using random features since we don't have node features)
        num_nodes = len(self.node_map)
        x = torch.randn((num_nodes, 16))  # 16-dimensional random features
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)

class EdgePredictorGNN(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        # Node embeddings
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        
        # Edge prediction
        src, dst = data.edge_index
        h_src = x[src]
        h_dst = x[dst]
        
        # Combine node embeddings and edge features
        edge_emb = torch.cat([h_src, h_dst, data.edge_attr], dim=1)
        return self.edge_predictor(edge_emb)

def train():
    # Load and preprocess data
    df = pd.read_csv('data2.csv')
    

    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9  # Convert to seconds


    # Prepare features and target
    features = ['throughput',  'new_request','istio_request_bytes','average_latency','istio_request_bytes','request_rate']  # Excluding error_rate
    target = 'status'
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    #df[features] = StandardScaler().fit_transform(df[features])
    
    # Create node mapping
    all_nodes = sorted(list(set(df['source_workload'].unique()).union(set(df['destination_workload'].unique()))))
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    # Create dataset
    dataset = MicroserviceDataset(df, node_map, features, target)
    
    # Since we have one graph, we'll use a single DataLoader with batch_size=1
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Initialize model
    model = EdgePredictorGNN(
        node_features=16,  # Matches our random node features
        edge_features=len(features),
        hidden_dim=64,
        num_classes=2  # healthy, degraded, error
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Print training stats
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d} | Loss: {total_loss/len(loader):.4f}')
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        data = dataset[0].to(device)
        out = model(data)
        pred = out.argmax(dim=1).cpu().numpy()
        truth = data.y.cpu().numpy()
        
        print("\nClassification Report:")
        print(classification_report(truth, pred, target_names=['healthy', 'degraded']))
        print("Macro F1-score:", f1_score(truth, pred, average='macro'))

if __name__ == "__main__":
    train()