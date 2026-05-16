import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.preprocess import convert_stl_to_aligned_voxels
from src.model import CdValuePredictor3D

# =========================================================
# MASTER CONTROL CONSTANT
# Change this single variable to scale between 64 and 128 pipeline
# =========================================================
GLOBAL_RESOLUTION = 64 # Toggle between 64 and 128 depending on your compute targets

class VoxelCdDataset(Dataset):
    def __init__(self, csv_file, stl_dir, voxel_dir, resolution=GLOBAL_RESOLUTION):
        self.meta_data = pd.read_csv(csv_file)
        self.stl_dir = stl_dir
        self.voxel_dir = voxel_dir
        self.resolution = resolution

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        row = self.meta_data.iloc[idx]
        stl_name = row['filename']
        true_cd = float(row['cd_value'])
        
        # Suffix the cache with resolution size to prevent grid mismatches
        voxel_name = stl_name.replace('.stl', f'_{self.resolution}res.npy')
        voxel_path = os.path.join(self.voxel_dir, voxel_name)
        
        if os.path.exists(voxel_path):
            voxel_matrix = np.load(voxel_path)
        else:
            stl_path = os.path.join(self.stl_dir, stl_name)
            voxel_matrix = convert_stl_to_aligned_voxels(stl_path, resolution=self.resolution)
            np.save(voxel_path, voxel_matrix)
            
        voxel_tensor = torch.tensor(voxel_matrix, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor([true_cd], dtype=torch.float32)
        
        return voxel_tensor, target_tensor

def main():
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for training: {device} | Active Pipeline Resolution: {GLOBAL_RESOLUTION}³")
    
    dataset = VoxelCdDataset(
        csv_file="dataset_meta.csv",
        stl_dir="data/raw_stl",
        voxel_dir="data/processed_voxels",
        resolution=GLOBAL_RESOLUTION
    )
    
    current_batch = min(BATCH_SIZE, len(dataset))
    dataloader = DataLoader(dataset, batch_size=current_batch, shuffle=True)
    
    model = CdValuePredictor3D(input_resolution=GLOBAL_RESOLUTION).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []
    
    print("\nStarting training loop...")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=True)
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            loop.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(dataset)
        loss_history.append(epoch_loss)
        
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/cd_predictor_3dcnn_{GLOBAL_RESOLUTION}.pth")
    print(f"\nTraining complete. Saved model weights to 'models/cd_predictor_3dcnn_{GLOBAL_RESOLUTION}.pth'.")

    # --- Visualizations ---
    print("\nGenerating training loss curve plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), loss_history, label='Training Loss', color='blue', linewidth=2)
    plt.title(f'3D CNN Aerodynamic Model - Training Loss Curve ({GLOBAL_RESOLUTION}³)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/loss_history.png", dpi=300)
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
    plt.scatter(all_targets, all_preds, color='crimson', alpha=0.7, edgecolors='k', label='Predicted Cars')
    min_val = min(min(all_targets), min(all_preds)) * 0.95
    max_val = max(max(all_targets), max(all_preds)) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='Perfect Prediction (y=x)')
    plt.title(f'Prediction Accuracy - Aerodynamic Drag ({GLOBAL_RESOLUTION}³)')
    plt.xlabel('True Cd Value')
    plt.ylabel('Predicted Cd Value')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig("outputs/prediction_accuracy.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()