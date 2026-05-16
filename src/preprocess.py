import trimesh
import numpy as np

def convert_stl_to_aligned_voxels(stl_path, resolution=64):
    """
    Loads an STL file, applies optimal alignments for aerodynamics (Cd) prediction,
    and converts it into a fixed-size cubic voxel grid.
    """
    mesh = trimesh.load(stl_path)
    
    bounds = mesh.bounds  
    extents = mesh.extents  
    
    # Rigid alignment for ground clearance (Z-axis) and symmetry (Y-axis)
    current_center = mesh.bounding_box.center_mass
    translation = [
        -current_center[0],  
        -current_center[1],  
        -bounds[0][2]        
    ]
    mesh.apply_translation(translation)
    
    # Scale normalization fitting within 95% of the total grid boundaries
    max_len = np.max(extents)
    target_size = resolution * 0.95  
    scale_factor = target_size / max_len
    mesh.apply_scale(scale_factor)
    
    # Voxelization with pitch=1.0 for unit cell grids
    voxels = mesh.voxelized(pitch=1.0)
    raw_matrix = voxels.matrix.astype(np.float32)
    
    # Anchor and embed into fixed container
    final_voxels = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    r_x, r_y, r_z = raw_matrix.shape
    
    crop_x = min(r_x, resolution)
    crop_y = min(r_y, resolution)
    crop_z = min(r_z, resolution)
    
    start_x = (resolution - crop_x) // 2
    start_y = (resolution - crop_y) // 2
    start_z = 0  
    
    final_voxels[
        start_x : start_x + crop_x,
        start_y : start_y + crop_y,
        start_z : start_z + crop_z
    ] = raw_matrix[:crop_x, :crop_y, :crop_z]
    
    return final_voxels