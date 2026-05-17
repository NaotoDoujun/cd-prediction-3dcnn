import os
import argparse
import torch
import numpy as np
from src.preprocess import convert_stl_to_aligned_voxels
from src.model import CdValuePredictor3D

def predict_cd(stl_path, model_path=None, resolution=64):
    """
    Loads a trained 3D CNN model, preprocesses a new STL file, 
    and predicts its Aerodynamic Drag Coefficient (Cd value).
    This core inference computation strictly operates independently of Open3D.
    """
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"Target STL file not found: {stl_path}")
    
    # Automatically resolve default weight path names based on selected resolution
    if model_path is None:
        model_path = f"models/cd_predictor_3dcnn_{resolution}.pth"
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model weights not found: {model_path}")

    # Optimized hardware accelerator setup targeting Apple Silicon MPS pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for inference: {device} | Targeted Resolution: {resolution}³")

    print(f"Preprocessing geometry: {stl_path}...")
    voxel_matrix = convert_stl_to_aligned_voxels(stl_path, resolution=resolution)
    
    # Format raw array into a 5D Tensor matching standard 3D CNN Batch/Channel criteria
    voxel_tensor = torch.tensor(voxel_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    voxel_tensor = voxel_tensor.to(device)

    print(f"Loading trained weights from: {model_path}...")
    model = CdValuePredictor3D(input_resolution=resolution).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  

    print("Running aerodynamic prediction...")
    with torch.no_grad():  
        prediction = model(voxel_tensor)
        predicted_cd = prediction.item()

    return predicted_cd, voxel_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Cd value from a 3D STL file.")
    parser.add_argument("--stl", type=str, required=True, help="Path to the input STL file")
    parser.add_argument("--model", type=str, default=None, help="Path to specific model weights (Optional)")
    parser.add_argument("--res", type=int, default=64, help="Voxel resolution matching the trained model (e.g. 64 or 128)")
    parser.add_argument("--visualize", action="store_true", help="Render the preprocessed voxel grid in 3D using Open3D")
    args = parser.parse_args()

    try:
        # Core execution does not invoke or require Open3D
        cd_result, voxel_matrix = predict_cd(stl_path=args.stl, model_path=args.model, resolution=args.res)
        print("\n" + "="*40)
        print(f"Target File: {args.stl}")
        print(f"Resolution:  {args.res}x{args.res}x{args.res}")
        print(f"Predicted Aerodynamic Drag Coefficient (Cd): {cd_result:.4f}")
        print("="*40)
        
        if args.visualize:
            # Safely handle lazy loading of the visualization layer
            try:
                import open3d as o3d
            except ImportError:
                print("\n[Error] Open3D is not installed in the current environment.")
                print("To unlock 3D rendering capabilities, please install it using:")
                print("    pip install open3d")
                print("Proceeding without 3D visualization preview.\n")
                exit(0)

            print("\nTriggering Solid Mesh + Voxel Overlay Preview...")
            
            # 1. Prepare Voxel Grid Geometry
            indices = np.argwhere(voxel_matrix > 0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(indices)
            
            # Cool semi-transparent feeling Blue for Voxels
            color_array = np.tile([0.2, 0.4, 0.8], (len(indices), 1))
            pcd.colors = o3d.utility.Vector3dVector(color_array)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
            
            # 2. Load and Apply Exact Pipeline Math to the Original Mesh for Overlay Alignment
            orig_mesh = o3d.io.read_triangle_mesh(args.stl)
            orig_mesh.compute_vertex_normals()
            
            # Mirror the translation logic from src/preprocess.py
            bounds = orig_mesh.get_axis_aligned_bounding_box()
            min_bound = bounds.get_min_bound()
            center_mass = orig_mesh.get_center()
            
            # Shift translation: center X, center Y, floor Z to 0
            t_vector = np.array([-center_mass[0], -center_mass[1], -min_bound[2]])
            orig_mesh.translate(t_vector)
            
            # Mirror the scaling factor logic from src/preprocess.py
            extents = bounds.get_max_bound() - bounds.get_min_bound()
            max_len = np.max(extents)
            target_size = args.res * 0.95
            scale_factor = target_size / max_len
            orig_mesh.scale(scale_factor, center=np.array([0, 0, 0]))
            
            # Center alignment container shifting inside the Res³ bounding box
            mesh_bounds = orig_mesh.get_axis_aligned_bounding_box()
            mesh_min = mesh_bounds.get_min_bound()
            mesh_max = mesh_bounds.get_max_bound()
            mesh_size = mesh_max - mesh_min
            
            crop_x = min(mesh_size[0], args.res)
            crop_y = min(mesh_size[1], args.res)
            
            start_x = (args.res - crop_x) // 2
            start_y = (args.res - crop_y) // 2
            
            # Final alignment injection shift
            final_shift = np.array([start_x - mesh_min[0], start_y - mesh_min[1], 0.0])
            orig_mesh.translate(final_shift)
            
            # Paint the CAD mesh Crimson Red so it acts as the baseline shape reference
            orig_mesh.paint_uniform_color([0.8, 0.2, 0.2])
            
            # 3. Create a Custom Visualizer to activate mesh styling features safely
            vis = o3d.visualization.Visualizer()
            window_title = f"3D CNN Voxel Grid & CAD Mesh Overlay Preview ({args.res}³ Resolution)"
            vis.create_window(window_name=window_title, width=1152, height=864)
            
            # Add all elements to the renderer
            vis.add_geometry(voxel_grid)
            vis.add_geometry(orig_mesh)
            
            # 4. Generate Coordinate System Axis
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.res * 0.2)
            vis.add_geometry(axes)
            
            # 5. Safely force standard Wireframe overlay on the red mesh
            render_option = vis.get_render_option()
            render_option.mesh_show_wireframe = True
            
            print("\n--- Interactive Fusion Viewer Initialized ---")
            print("  - Red Solid + Grids : Original CAD Vehicle Silhouette (The Target Shape)")
            print("  - Blue Cubes        : Preprocessed 3D CNN Input Grid")
            print("Close the visualizer window to resume script execution.\n")
            
            vis.run()
            vis.destroy_window()
            
    except Exception as e:
        print(f"\nError during inference or visualization rendering: {e}")