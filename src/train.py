import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
    print(f"Using device: {device}")
    
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
    
    print("\nStarting training loop...")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        # Wrap dataloader with tqdm for progress tracking
        loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=True)
        
        for inputs, targets in loop:
            # Transfer batch data to designated hardware (MPS or CPU)
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
            
        epoch_loss = running_loss / len(dataset)
        
    # Save trained model weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cd_predictor_3dcnn.pth")
    print("\nTraining complete. Saved model state dictionary to 'models/cd_predictor_3dcnn.pth'.")

if __name__ == "__main__":
    main()