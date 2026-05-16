import torch
import torch.nn as nn

class CdValuePredictor3D(nn.Module):
    def __init__(self, input_resolution=64):
        super(CdValuePredictor3D, self).__init__()
        self.resolution = input_resolution
        
        # 3D Convolutional layers
        # Expected input shape: [Batch_Size, Channels(1), Res, Res, Res]
        self.features = nn.Sequential(
            # Layer 1: Res -> Res/2
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Layer 2: Res/2 -> Res/4
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Layer 3: Res/4 -> Res/8
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Layer 4: Res/8 -> Res/16
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # 4 MaxPool3d layers divide spatial dimensions by 16 (2^4)
        final_spatial_dim = input_resolution // 16
        flat_features = 128 * (final_spatial_dim ** 3)
        
        # Fully connected layers for regression
        self.regressor = nn.Sequential(
            nn.Linear(flat_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Prevents overfitting
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)    # Final continuous prediction output
        )

    def forward(self, x):
        # x shape: [Batch, 1, Res, Res, Res]
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # Flatten spatial dimensions
        x = self.regressor(x)
        return x

if __name__ == "__main__":
    res = 128  # 👈 Easily verifiable test for 128 resolution
    model = CdValuePredictor3D(input_resolution=res)
    print(f"Model architecture built successfully for resolution: {res}")
    
    # Create dummy tensor [Batch=2, Channels=1, 128, 128, 128]
    dummy_input = torch.randn(2, 1, res, res, res)
    
    # Inference test
    output = model(dummy_input)
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape} (Expected: [2, 1])")