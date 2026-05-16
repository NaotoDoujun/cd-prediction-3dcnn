import trimesh
import numpy as np

def convert_stl_to_aligned_voxels(stl_path, resolution=64):
    """
    Loads an STL file, applies optimal alignments for aerodynamics (Cd) prediction,
    and converts it into a fixed-size cubic voxel grid.
    
    Parameters:
        stl_path (str): Path to the input STL file.
        resolution (int): Resolution along one side of the cubic grid (e.g., 64, 128).
    
    Returns:
        np.ndarray: A float32 array of shape (resolution, resolution, resolution) containing 0.0 or 1.0.
    """
    # 1. Load STL file
    mesh = trimesh.load(stl_path)
    
    # ---------------------------------------------------------
    # Approach 1: Standardize orientation (Direction of travel)
    # Assumes the CATIA export is aligned as "X-axis = Forward, Z-axis = Up".
    # If rotation is needed, apply mesh.apply_transform() here.
    # ---------------------------------------------------------
    
    # 2. Get geometry metadata
    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    extents = mesh.extents  # Dimensions along each axis [length_x, width_y, height_z]
    
    # ---------------------------------------------------------
    # Approach 2: Rigid alignment for ground (Z-axis) and symmetry (Y-axis)
    # To maintain the critical ground clearance effect, fix the lowest point (Z-min) 
    # to Z=0, and center the lateral axis (Y-center) to Y=0.
    # ---------------------------------------------------------
    current_center = mesh.bounding_box.center_mass
    
    translation = [
        -current_center[0],  # Center the X-axis (Lengthwise) at origin
        -current_center[1],  # Center the Y-axis (Widthwise) at origin for symmetry
        -bounds[0][2]        # Anchor the lowest point (Z-min) to Z=0
    ]
    mesh.apply_translation(translation)
    
    # ---------------------------------------------------------
    # Approach 3: Scale normalization (Fitting into bounding box)
    # Scale based on the longest dimension (usually X-axis length) to fit within the grid.
    # Apply a small margin (5%) to prevent clipping artifacts at the boundaries.
    # ---------------------------------------------------------
    max_len = np.max(extents)
    target_size = resolution * 0.95  # Fit within 95% of the total grid size
    scale_factor = target_size / max_len
    mesh.apply_scale(scale_factor)
    
    # ---------------------------------------------------------
    # 4. Perform voxelization
    # Set pitch to 1.0 to treat each voxel cell as a standard unit cube
    # ---------------------------------------------------------
    voxels = mesh.voxelized(pitch=1.0)
    
    # Extract raw binary voxel matrix
    raw_matrix = voxels.matrix.astype(np.float32)
    
    # ---------------------------------------------------------
    # Approach 4: Padding/Embedding into a fixed-size 3D array (Tensor)
    # Since trimesh dynamically changes matrix shape depending on object bounds,
    # embed the raw matrix into a fixed (Res, Res, Res) container for the AI model.
    # ---------------------------------------------------------
    final_voxels = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    
    # Calculate dimensions of the raw matrix
    r_x, r_y, r_z = raw_matrix.shape
    
    # Safe clipping in case normalized dimensions slightly exceed the resolution limit
    crop_x = min(r_x, resolution)
    crop_y = min(r_y, resolution)
    crop_z = min(r_z, resolution)
    
    # Define start indices (Center X/Y axes, keep Z anchored at index 0 for ground contact)
    start_x = (resolution - crop_x) // 2
    start_y = (resolution - crop_y) // 2
    start_z = 0  
    
    # Copy raw matrix into the fixed container
    final_voxels[
        start_x : start_x + crop_x,
        start_y : start_y + crop_y,
        start_z : start_z + crop_z
    ] = raw_matrix[:crop_x, :crop_y, :crop_z]
    
    return final_voxels

if __name__ == "__main__":
    resolution_size = 64
    voxel_data = convert_stl_to_aligned_voxels("catia_output_model.stl", resolution=resolution_size)
    
    print(f"Conversion complete. Array shape: {voxel_data.shape}")
    print(f"Total voxel grid size: {voxel_data.size}")
    print(f"Solid (vehicle body) voxel count: {np.sum(voxel_data)}")
    
    # Save processed array to disk for model training
    np.save("aligned_car_voxel.npy", voxel_data)