import os
import argparse
import torch
from src.preprocess_pointnet import convert_stl_to_aligned_pointcloud
from src.model_pointnet import CdPredictorPointNet

def predict_pointnet_cd(stl_path, model_path=None, num_points=2048):
    """
    Loads a trained PointNet++ model, samples an aligned point cloud from the input STL,
    and predicts the Aerodynamic Drag Coefficient (Cd value).
    Core inference operates independently of Open3D.
    """
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"Target STL file not found: {stl_path}")
    
    if model_path is None:
        model_path = f"models/cd_predictor_pointnet_{num_points}.pth"
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model weights not found: {model_path}")

    # Device selection matching Apple Silicon MPS / CUDA / CPU pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for PointNet++ inference: {device} | Target Density: {num_points} pts")

    print(f"Preprocessing geometry into point cloud: {stl_path}...")
    points = convert_stl_to_aligned_pointcloud(stl_path, num_points=num_points)
    
    # Format raw array into a 3D Tensor [Batch=1, Points=2048, XYZ=3]
    points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)

    print(f"Loading trained PointNet++ weights from: {model_path}...")
    model = CdPredictorPointNet(num_points=num_points).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  

    print("Running aerodynamic prediction via PointNet++...")
    with torch.no_grad():  
        prediction = model(points_tensor)
        predicted_cd = prediction.item()

    return predicted_cd, points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Cd value from a 3D STL file using PointNet++.")
    parser.add_argument("--stl", type=str, required=True, help="Path to the input STL file")
    parser.add_argument("--model", type=str, default=None, help="Path to specific model weights (Optional)")
    parser.add_argument("--points", type=int, default=2048, help="Number of points matching the trained model")
    parser.add_argument("--visualize", action="store_true", help="Render the point cloud overlayed with the original CAD mesh")
    args = parser.parse_args()

    try:
        # 1. Run inference
        cd_result, sampled_points = predict_pointnet_cd(stl_path=args.stl, model_path=args.model, num_points=args.points)
        print("\n" + "="*50)
        print(f"Target PointNet++ File:                       {args.stl}")
        print(f"Sampled Density:                              {args.points} pts")
        print(f"Predicted Aerodynamic Drag Coefficient (Cd): {cd_result:.4f}")
        print("="*50)
        
        # 2. Trigger visualization overlay via centralized toolbox modules
        if args.visualize:
            try:
                from src.visualize_pointnet import show_pointnet_overlay_with_open3d
            except ImportError:
                print("\n[Error] open3d and trimesh are required for visualization.")
                exit(0)
                
            show_pointnet_overlay_with_open3d(
                stl_path=args.stl, 
                sampled_points=sampled_points, 
                num_points=args.points
            )
            
    except Exception as e:
        print(f"\nError during inference or visualization rendering: {e}")