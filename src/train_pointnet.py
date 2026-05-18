import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.preprocess_pointnet import convert_stl_to_aligned_pointcloud
from src.model_pointnet import CdPredictorPointNet

# =========================================================
# MASTER CONTROL CONSTANT
# Change this single variable to scale between 2048, 4096, or 8192 pipeline density
# =========================================================
GLOBAL_POINTS = 4096  # Target point cloud sampling density context

class PointCloudCdDataset(Dataset):
    def __init__(self, csv_file, stl_dir, pcd_dir, num_points=GLOBAL_POINTS):
        self.meta_data = pd.read_csv(csv_file)
        self.stl_dir = stl_dir
        self.pcd_dir = pcd_dir
        self.num_points = num_points

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        row = self.meta_data.iloc[idx]
        stl_name = row['filename']
        true_cd = float(row['cd_value'])
        
        # Suffix the cache with point count sizes to prevent grid mismatches
        pcd_name = stl_name.replace('.stl', f'_{self.num_points}pts.npy')
        pcd_path = os.path.join(self.pcd_dir, pcd_name)
        
        if os.path.exists(pcd_path):
            points = np.load(pcd_path)
        else:
            stl_path = os.path.join(self.stl_dir, stl_name)
            points = convert_stl_to_aligned_pointcloud(stl_path, num_points=self.num_points)
            os.makedirs(self.pcd_dir, exist_ok=True)
            np.save(pcd_path, points)
            
        return torch.tensor(points, dtype=torch.float32), torch.tensor([true_cd], dtype=torch.float32)

if __name__ == "__main__":
    # Project Paths Definition Matrix
    csv_metadata = "dataset_meta.csv"
    stl_directory = "data/raw_stl"
    pcd_cache_directory = "data/processed_pcd"
    
    # Accelerated processing unit matrix validation (Apple Silicon MPS optimization fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing PointNet++ Pipeline | Device: {device} | Point Density Target: {GLOBAL_POINTS}")

    # Dataset & DataLoader instantiation
    dataset = PointCloudCdDataset(csv_file=csv_metadata, stl_dir=stl_directory, pcd_dir=pcd_cache_directory, num_points=GLOBAL_POINTS)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model using the dynamic master control specs
    model = CdPredictorPointNet(num_points=GLOBAL_POINTS).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 40
    loss_history = []

    print("\nStarting PointNet++ Training Execution Loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.5f}"})
            
        epoch_loss = running_loss / len(dataset)
        loss_history.append(epoch_loss)
        print(f">> Epoch Complete Loss: {epoch_loss:.5f}\n")

    # Save weights with customized naming context matching standard 3D CNN patterns
    os.makedirs("models", exist_ok=True)
    weight_output_path = f"models/cd_predictor_pointnet_{GLOBAL_POINTS}.pth"
    torch.save(model.state_dict(), weight_output_path)
    print(f"Training finalized. Network weights exported safely to: {weight_output_path}")

    # Graphical analytical performance tracing plots
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', color='teal', label='Training Loss')
    plt.title(f'PointNet++ Aerodynamic Model - Training Loss Curve ({GLOBAL_POINTS} pts)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/loss_history_pointnet.png", dpi=300)
    plt.close()

    print("\nGenerating Prediction vs. True Value scatter plot...")
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
            
    plt.figure(figsize=(7, 7))
    plt.scatter(all_targets, all_preds, color='teal', alpha=0.7, edgecolors='k', label='Predicted Cars')
    min_val = min(min(all_targets), min(all_preds)) * 0.95
    max_val = max(max(all_targets), max(all_preds)) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='Perfect Prediction (y=x)')
    plt.title(f'Prediction Accuracy - PointNet++ ({GLOBAL_POINTS} pts)')
    plt.xlabel('True Cd Value')
    plt.ylabel('Predicted Cd Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("outputs/prediction_accuracy_pointnet.png", dpi=300)
    plt.close()
    print("Analytical diagnostics complete. Validation graphics saved inside /outputs directory.")