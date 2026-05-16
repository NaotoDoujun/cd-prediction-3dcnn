import os
import argparse
import torch
from src.preprocess import convert_stl_to_aligned_voxels
from src.model import CdValuePredictor3D

def predict_cd(stl_path, model_path="models/cd_predictor_3dcnn.pth", resolution=64):
    """
    Loads a trained 3D CNN model, preprocesses a new STL file, 
    and predicts its Aerodynamic Drag Coefficient (Cd value).
    """
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"Target STL file not found: {stl_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model weights not found: {model_path}")

    # 1. Device configuration (Leverages NVIDIA CUDA or Apple Silicon MPS if available, otherwise defaults to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for inference: {device}")

    # 2. Preprocess the input STL geometry into a fixed-size voxel grid
    print(f"Preprocessing geometry: {stl_path}...")
    voxel_matrix = convert_stl_to_aligned_voxels(stl_path, resolution=resolution)
    
    # Convert to PyTorch tensor and add Batch and Channel dimensions -> [1, 1, Res, Res, Res]
    voxel_tensor = torch.tensor(voxel_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    voxel_tensor = voxel_tensor.to(device)

    # 3. Load the trained model architecture and weights
    print(f"Loading trained weights from: {model_path}...")
    model = CdValuePredictor3D(input_resolution=resolution).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode

    # 4. Perform forward pass (Inference)
    print("Running aerodynamic prediction...")
    with torch.no_grad():  # Disable gradient computation for speed and memory efficiency
        prediction = model(voxel_tensor)
        predicted_cd = prediction.item()

    return predicted_cd

if __name__ == "__main__":
    # Set up argument parser for command line execution
    parser = argparse.ArgumentParser(description="Predict Cd value from a 3D STL file.")
    parser.add_argument("--stl", type=str, required=True, help="Path to the input STL file")
    parser.add_argument("--model", type=str, default="models/cd_predictor_3dcnn.pth", help="Path to the trained model weights")
    args = parser.parse_args()

    try:
        cd_result = predict_cd(stl_path=args.stl, model_path=args.model)
        print("\n" + "="*40)
        print(f"Target File: {args.stl}")
        print(f"Predicted Aerodynamic Drag Coefficient (Cd): {cd_result:.4f}")
        print("="*40)
    except Exception as e:
        print(f"\nError during inference: {e}")