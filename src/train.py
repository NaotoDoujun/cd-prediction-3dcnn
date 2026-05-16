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

# --- 1. Custom PyTorch Dataset Definition ---
class VoxelCdDataset(Dataset):
    def __init__(self, csv_file, stl_dir, voxel_dir, resolution=64):
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
        
        # Voxel matrix cache file (.npy)
        voxel_name = stl_name.replace('.stl', '.npy')
        voxel_path = os.path.join(self.voxel_dir, voxel_name)
        
        # Load preprocessed voxel if cached; otherwise, compute and cache it on the fly
        if os.path.exists(voxel_path):
            voxel_matrix = np.load(voxel_path)
        else:
            stl_path = os.path.join(self.stl_dir, stl_name)
            voxel_matrix = convert_stl_to_aligned_voxels(stl_path, resolution=self.resolution)
            np.save(voxel_path, voxel_matrix)
            
        # Convert to PyTorch tensor with shape [Channels(1), H, W, D]
        voxel_tensor = torch.tensor(voxel_matrix, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor([true_cd], dtype=torch.float32)
        
        return voxel_tensor, target_tensor

# --- 2. Main Training Loop ---
def main():
    # Hyperparameters
    RESOLUTION = 64
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # 1. Device configuration (Leverages NVIDIA CUDA or Apple Silicon MPS if available, otherwise defaults to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for training: {device}")
    
    # Initialize dataset and data loader
    dataset = VoxelCdDataset(
        csv_file="dataset_meta.csv",
        stl_dir="data/raw_stl",
        voxel_dir="data/processed_voxels",
        resolution=RESOLUTION
    )
    
    # Adjust batch size if the dataset size is smaller than the default configuration
    current_batch = min(BATCH_SIZE, len(dataset))
    dataloader = DataLoader(dataset, batch_size=current_batch, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = CdValuePredictor3D(input_resolution=RESOLUTION).to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # List to store loss tracking history
    loss_history = []
    
    print("\nStarting training loop...")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        # Wrap dataloader with tqdm for progress tracking
        loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=True)
        
        for inputs, targets in loop:
            # Transfer batch data to designated hardware (CUDA, MPS, or CPU)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Clear gradients from previous step
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass (Compute gradients)
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            loop.set_postfix(loss=loss.item())
            
        # Calculate average loss for the current epoch
        epoch_loss = running_loss / len(dataset)
        loss_history.append(epoch_loss)
        
    # Save trained model weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cd_predictor_3dcnn.pth")
    print("\nTraining complete. Saved model state dictionary to 'models/cd_predictor_3dcnn.pth'.")

    # ---------------------------------------------------------
    # Visualization 1: Generate Training Loss Curve
    # ---------------------------------------------------------
    print("\nGenerating training loss curve plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), loss_history, label='Training Loss', color='blue', linewidth=2)
    plt.title('3D CNN Aerodynamic Model - Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save loss history graph
    os.makedirs("outputs", exist_ok=True)
    plot_path = "outputs/loss_history.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()  # Clear current figure memory
    print(f"Successfully saved loss plot to: {plot_path}")

    # ---------------------------------------------------------
    # Visualization 2: Generate Predicted vs. True Value Scatter Plot
    # ---------------------------------------------------------
    print("\nGenerating Prediction vs. True Value scatter plot...")
    model.eval()
    all_preds = []
    all_targets = []
    
    # Aggregate prediction values across the entire dataset
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
            
    plt.figure(figsize=(7, 7))
    # Plot empirical data points
    plt.scatter(all_targets, all_preds, color='crimson', alpha=0.7, edgecolors='k', label='Predicted Cars')
    
    # Render the reference line (y = x) representing ideal prediction accuracy
    min_val = min(min(all_targets), min(all_preds)) * 0.95
    max_val = max(max(all_targets), max(all_preds)) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='Perfect Prediction (y=x)')
    
    plt.title('Prediction Accuracy - Aerodynamic Drag (Cd)')
    plt.xlabel('True Cd Value (CFD / Experimental)')
    plt.ylabel('Predicted Cd Value (AI Output)')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Save accuracy verification scatter plot
    scatter_path = "outputs/prediction_accuracy.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()  # Clear current figure memory
    print(f"Successfully saved scatter plot to: {scatter_path}")

if __name__ == "__main__":
    main()