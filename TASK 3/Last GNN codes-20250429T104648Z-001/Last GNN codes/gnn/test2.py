import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric_temporal.nn.recurrent import GCLSTM
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TemporalMicroserviceDataset:
    def __init__(self, df, node_map, features, target):
        self.df = df
        self.node_map = node_map
        self.features = features
        self.target = target
        self.time_windows = sorted(df['time_window'].unique())
        self.num_nodes = len(node_map)
    
    def __len__(self):
        return len(self.time_windows) - 1
    
    def __getitem__(self, idx):
        current_window = self.time_windows[idx]
        next_window = self.time_windows[idx+1]
        
        current_data = self.df[self.df['time_window'] == current_window]
        next_data = self.df[self.df['time_window'] == next_window]
        
        # Get all active nodes in this time window
        active_nodes = set(current_data['source_workload']).union(set(current_data['destination_workload']))
        active_nodes.update(set(next_data['source_workload']).union(set(next_data['destination_workload'])))
        
        # Create mapping from global node indices to local indices
        local_node_map = {node: i for i, node in enumerate(sorted(active_nodes))}
        
        edge_index = []
        edge_attrs = []
        edge_labels = []
        
        for _, row in next_data.iterrows():
            src = local_node_map[row['source_workload']]
            tgt = local_node_map[row['destination_workload']]
            edge_index.append([src, tgt])
            edge_attrs.append(row[self.features].values)
            edge_labels.append(row[self.target])
        
        if not edge_index:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attrs = torch.empty((0, len(self.features)), dtype=torch.float)
            edge_labels = torch.empty((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
            edge_labels = torch.tensor(edge_labels, dtype=torch.long)
        
        # Create node features only for active nodes
        x = torch.randn((len(local_node_map), 16))
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attrs,
            y=edge_labels,
            local_node_map=local_node_map  # Store mapping for debugging
        )

class TemporalGNN(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, num_classes):
        super().__init__()
        self.recurrent = GCLSTM(
            in_channels=node_features,
            out_channels=hidden_dim,
            K=1
        )
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.h = None
        self.c = None
        
    def forward(self, data):
        # Add batch dimension [1, num_nodes, node_features]
        x = data.x.unsqueeze(0)
        
        # Pass through GCLSTM
        h, c = self.recurrent(x, data.edge_index, self.h, self.c)
        self.h, self.c = h.detach(), c.detach()
        
        # Remove batch dimension [num_nodes, hidden_dim]
        h = h.squeeze(0)
        
        # Get source and target node embeddings
        src, dst = data.edge_index
        h_src = h[src]
        h_dst = h[dst]
        
        # Ensure edge_attr has correct shape
        edge_attr = data.edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        
        # Combine features and predict
        edge_emb = torch.cat([h_src, h_dst, edge_attr], dim=1)
        return self.edge_predictor(edge_emb)

def train_model():
    # Load and preprocess data
    df = pd.read_csv('data_file.csv', delimiter='\t')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_window'] = df['timestamp'].dt.floor('1min')
    
    features = ['throughput', 'average_latency']
    target = 'status'
    df[features] = StandardScaler().fit_transform(df[features])
    
    # Create global node mapping
    all_nodes = sorted(list(set(df['source_workload'].unique()).union(set(df['destination_workload'].unique()))))
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    # Create dataset
    dataset = TemporalMicroserviceDataset(df, node_map, features, target)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset = [dataset[i] for i in range(train_size)]
    val_dataset = [dataset[i] for i in range(train_size, train_size + val_size)]
    test_dataset = [dataset[i] for i in range(train_size + val_size, len(dataset))]
    
    # Initialize model
    model = TemporalGNN(
        node_features=16,
        edge_features=len(features),
        hidden_dim=64,
        num_classes=3
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_f1 = 0
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        processed_batches = 0
        
        for data in train_dataset:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Skip empty graphs
            if data.edge_index.size(1) == 0:
                continue
                
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            processed_batches += 1
        
        # Validation
        model.eval()
        val_preds, val_truths = [], []
        with torch.no_grad():
            for data in val_dataset:
                data = data.to(device)
                if data.edge_index.size(1) == 0:
                    continue
                out = model(data)
                val_preds.append(out.argmax(dim=1).cpu())
                val_truths.append(data.y.cpu())
        
        if val_preds:
            val_preds = torch.cat(val_preds).numpy()
            val_truths = torch.cat(val_truths).numpy()
            val_f1 = f1_score(val_truths, val_preds, average='macro')
            avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
            print(f'Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}')
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), 'best_model.pt')
    
    # Final evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    test_preds, test_truths = [], []
    with torch.no_grad():
        for data in test_dataset:
            data = data.to(device)
            if data.edge_index.size(1) == 0:
                continue
            out = model(data)
            test_preds.append(out.argmax(dim=1).cpu())
            test_truths.append(data.y.cpu())
    
    if test_preds:
        test_preds = torch.cat(test_preds).numpy()
        test_truths = torch.cat(test_truths).numpy()
        print("\nTest Results:")
        print(classification_report(test_truths, test_preds, target_names=['healthy', 'degraded', 'error']))
        print("Macro F1-score:", f1_score(test_truths, test_preds, average='macro'))
    else:
        print("No test predictions were generated - check your data")

if __name__ == "__main__":
    train_model()