import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


class VehicleWindowDataset(Dataset):
    def __init__(self, X, y, vids, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # shape (N,1)
        self.vids = vids
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

def train(X, y, vids,epochs=10):
    dataset = VehicleWindowDataset(X, y, vids)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = CNNRegressor(num_features=X.shape[2], window_size=X.shape[1], output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # simple training loop
    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in dataloader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return model

def predict_windows(model, X, batch_size=64, device=None):
    """
    Run inference on windowed data (X, optionally y, vids) and return predictions.
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained model (should be in eval mode).
    X : np.ndarray or torch.Tensor
        Array of shape (N, window_size, num_features).
    y : np.ndarray or torch.Tensor, optional
        True targets (for optional comparison), shape (N,) or (N,1).
    vids : array-like, optional
        Vehicle IDs corresponding to each sample/window, length N.
    batch_size : int
        Batch size for inference.
    device : torch.device or str, optional
        Device to run inference on (e.g. “cuda” or “cpu”). If None, defaults to model’s device or “cpu”.
    
    Returns
    -------
    y_preds : np.ndarray of shape (N,)
        Predicted values (flattened if multiple dims).
    y_true : np.ndarray or None
        If `y` is provided, returns ground truth as array.
    vids_out : np.ndarray or None
        If `vids` is provided, returns the vehicle ID array.
    """
    # Convert to tensor if needed
    if not torch.is_tensor(X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X.float()
    # Move to device
    if device is None:
        device = next(model.parameters()).device
    X_tensor = X_tensor.to(device)
    
    # Prepare DataLoader (no shuffle)
    
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    y_preds_list = []
    # Optionally  y_true and vids
    
    with torch.no_grad():
        for batch in dataloader:
            (xb,) = batch
            preds = model(xb)  # shape (batch, output_dim)
            # If output_dim > 1, you might want different handling
            # Here we flatten to 1D if single output
            preds = preds.view(-1).cpu().numpy()
            y_preds_list.append(preds)
    
    # Concatenate batches
    y_preds = np.concatenate(y_preds_list, axis=0)
    
    return y_preds



def make_vehicle_windows(df, vehicle_id_col, feature_cols, target_col,
                         window_size, stride=1, repeat_initial=True):
    """
    Create sliding windows per vehicle.

    Parameters
    ----------
    df : pandas.DataFrame — contains vehicle_id_col, feature_cols, target_col
    vehicle_id_col : str — column name for vehicle identifier
    feature_cols : list[str] — list of feature column names
    target_col : str — name of target (e.g., RUL)
    window_size : int — number of time‐steps in each window
    stride : int — window stride (default 1)
    repeat_initial : bool — if True, the first (window_size-1) time‐steps of a vehicle
                            will be “repeated” (i.e., pad with first row) so that
                            you get a full first window.

    Returns
    -------
    X : np.ndarray of shape (num_windows, window_size, num_features)
    y : np.ndarray of shape (num_windows, )
    vehicle_ids : np.ndarray of length num_windows — vehicle id for each window
    """
    X_list = []
    y_list = []
    vid_list = []
    for vid, sub in df.groupby(vehicle_id_col):
        sub = sub.sort_index()  # or sort by time if you have a time index
        feat = sub[feature_cols].values
        targ = sub[target_col].values
        n = len(sub)
        if repeat_initial and n < window_size:
            # pad up to window_size by repeating first row
            pad = np.repeat(feat[0:1, :], window_size - n, axis=0)
            feat = np.vstack([pad, feat])
            targ = np.concatenate([np.repeat(targ[0], window_size - n), targ])
            n = window_size
        
        # Build windows
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            X_list.append(feat[start:end, :])
            # choose target at the *end* of window (could choose other strategy)
            y_list.append(targ[end - 1])
            vid_list.append(vid)
        
        # Optionally: you might also want to include “last few” windows if n-window_size not divisible by stride
        # up to you.
    X = np.stack(X_list)
    y = np.array(y_list)
    vehicle_ids = np.array(vid_list)
    return X, y, vehicle_ids

class CNNRegressor(nn.Module):
    def __init__(self, num_features, window_size, output_dim=1, 
                 conv_filters=[64,32], kernel_sizes=[3,3], 
                 dropout=0.2):
        super(CNNRegressor, self).__init__()
        assert len(conv_filters) == len(kernel_sizes)
        self.num_features = num_features
        self.window_size = window_size
        
        # 1D convolution layers: input shape (batch, features, time) for Conv1d
        layers = []
        in_ch = num_features
        for out_ch, k in zip(conv_filters, kernel_sizes):
            layers.append(nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                                    kernel_size=k, padding=k//2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        
        # compute size after conv+pooling
        # approximate: window_size → after each pool halved
        conv_time = window_size
        for _ in conv_filters:
            conv_time = conv_time // 2
        
        self.fc1 = nn.Linear(in_ch * conv_time, 128)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        # x: (batch, time_steps, num_features)
        # need to permute for Conv1d: (batch, features, time)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        out = self.fc2(x)
        return out
