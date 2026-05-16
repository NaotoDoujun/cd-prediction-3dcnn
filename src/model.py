import torch
import torch.nn as nn

class CdValuePredictor3D(nn.Module):
    def __init__(self, input_resolution=64):
        super(CdValuePredictor3D, self).__init__()
        self.resolution = input_resolution
        
        # 3D Convolutional layers
        # Expected input shape: [Batch_Size, Channels(1), Height(64), Width(64), Depth(64)]
        self.features = nn.Sequential(
            # Layer 1: 64x64x64 -> 32x32x32
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Layer 2: 32x32x32 -> 16x16x16
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Layer 3: 16x16x16 -> 8x8x8
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Layer 4: 8x8x8 -> 4x4x4
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Calculate flattened feature dimension prior to fully connected layers
        # With input_resolution=64, 4 MaxPool3d operations reduce the dimension to 4 (64 -> 32 -> 16 -> 8 -> 4)
        flat_features = 128 * (input_resolution // 16) ** 3
        
        # Fully connected layers for regression
        # Combines extracted spatial features to predict a single continuous target (Cd value)
        self.regressor = nn.Sequential(
            nn.Linear(flat_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Prevents overfitting
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)    # No activation function at the final layer for regression
        )

    def forward(self, x):
        # x shape: [Batch, 1, Res, Res, Res]
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # Flatten dimensions keeping the batch axis
        x = self.regressor(x)
        return x

if __name__ == "__main__":
    res = 64
    model = CdValuePredictor3D(input_resolution=res)
    print("Model architecture built successfully.")
    
    # Create dummy tensor [Batch=2, Channels=1, 64, 64, 64]
    dummy_input = torch.randn(2, 1, res, res, res)
    
    # Inference test
    output = model(dummy_input)
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape} (Expected: [2, 1])")
    print(f"Predicted values (initial state):\n{output}")