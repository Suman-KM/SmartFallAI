import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Add scripts directory to path for evaluate imports
sys.path.append(os.path.dirname(__file__))
from evaluate import compute_all_metrics, measure_inference_latency

class Conv1DNet(nn.Module):
    def __init__(self, in_channels, num_classes=14):
        super(Conv1DNet, self).__init__()
        # Input shape: (Batch, in_channels, 100)
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(2)  # (Batch, 64, 50)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        
        self.gap = nn.AdaptiveAvgPool1d(1)  # (Batch, 128, 1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x is (Batch, 100, in_channels), transpose to (Batch, in_channels, 100)
        x = x.transpose(1, 2)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = self.dropout1(out)
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.gap(out).squeeze(-1)
        out = self.fc(out)
        return out

def train_and_eval_cnn(X_train, y_train, X_val, y_val, out_dir, class_names, device_name, pipeline_id, epochs=25, batch_size=64, lr=0.001, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    in_channels = X_train.shape[2]
    num_classes = len(class_names)
    model = Conv1DNet(in_channels, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_val_f1 = -1.0
    best_model_weights = None
    training_history = []
    
    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            
        train_loss = total_loss / len(X_train)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_x_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_logits = model(val_x_tensor)
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            
        val_metrics = compute_all_metrics(y_val, val_preds, class_names)
        val_f1 = val_metrics["macro_f1"]
        
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_f1,
            "val_fall_recall": val_metrics["binary"]["fall_recall"]
        })
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_weights = model.state_dict().copy()
            
    train_duration = time.time() - start_train_time
    
    # Load best weights
    model.load_state_dict(best_model_weights)
    model.eval()
    
    # Final Validation Metrics
    with torch.no_grad():
        val_x_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_preds = torch.argmax(model(val_x_tensor), dim=1).cpu().numpy()
    final_val_metrics = compute_all_metrics(y_val, val_preds, class_names)
    
    # Measure Latency
    def predict_batch(x_arr):
        with torch.no_grad():
            t = torch.tensor(x_arr, dtype=torch.float32).to(device)
            return model(t).cpu().numpy()
            
    latency_ms = measure_inference_latency(predict_batch, X_val[:64])
    
    # Parameter count and Model Size
    total_params = sum(p.numel() for p in model.parameters())
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pth")
    torch.save(best_model_weights, model_path)
    model_size_kb = os.path.getsize(model_path) / 1024.0
    
    results = {
        "model_type": "1D_CNN",
        "device": device_name,
        "pipeline_id": pipeline_id,
        "feature_dim": in_channels,
        "total_parameters": total_params,
        "model_size_kb": model_size_kb,
        "train_duration_sec": train_duration,
        "inference_latency_ms": latency_ms,
        "training_history": training_history,
        "validation_metrics": final_val_metrics
    }
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    return results, model
