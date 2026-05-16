import os
import numpy as np
import trimesh
from src.preprocess import convert_stl_to_aligned_voxels
from src.train import GLOBAL_RESOLUTION

def test_pipeline():
    print("1. Generating a local 3D mesh (Sphere) for verification...")
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    
    # Shift geometry away from origin to simulate poor CAD exports and test rigid alignment
    mesh.apply_translation([10.0, -5.0, 8.5])
    
    test_stl_path = "data/raw_stl/test_sphere.stl"
    mesh.export(test_stl_path)
    print(f"   -> Saved misaligned test STL file to: {test_stl_path}")
    
    # Synchronize resolution size automatically with the training pipeline configuration
    resolution = GLOBAL_RESOLUTION
    print(f"\n2. Executing voxelization pipeline with rigid alignment (Resolution: {resolution})...")
    voxel_matrix = convert_stl_to_aligned_voxels(test_stl_path, resolution=resolution)
    
    print(f"   -> Conversion complete! Array shape: {voxel_matrix.shape}")
    print(f"   -> Filled voxel count (Solid object): {int(np.sum(voxel_matrix))}")
    
    # ---------------------------------------------------------
    # 3. Alignment Verification
    # ---------------------------------------------------------
    print("\n3. Validating spatial alignment matrices:")
    
    z_indices, y_indices, x_indices = np.where(voxel_matrix > 0)
    
    # Z-axis (Vertical): Check if anchored properly to index 0 (ground contact)
    min_z = np.min(z_indices)
    print(f"   [Z-axis (Vertical)] Minimum index: {min_z} (Expected: 0 -> Grounded contact verification)")
    
    # Y-axis / X-axis: Dynamically compute the center threshold based on target resolution
    expected_center = resolution / 2 - 0.5
    mean_y = np.mean(y_indices)
    print(f"   [Y-axis (Lateral)] Center of mass index: {mean_y:.2f} (Expected: ~{expected_center:.1f} -> Central symmetry check)")
    
    mean_x = np.mean(x_indices)
    print(f"   [X-axis (Longitudinal)] Center of mass index: {mean_x:.2f} (Expected: ~{expected_center:.1f} -> Central position check)")

    # ---------------------------------------------------------
    # 4. Terminal Voxel Visualizer (With Dynamic Auto-Downsampling)
    # ---------------------------------------------------------
    print("\n4. Displaying mid-plane cross-section (Side view profile):")
    mid_y = resolution // 2
    slice_side = voxel_matrix[:, mid_y, :]  # 2D cross-section array [Res, Res]
    
    # Scale down to standard 32 bins to prevent console character wrapping on higher resolutions
    vis_res = 32
    scale = max(1, resolution // vis_res)  # Evaluates to 2 for 64³ and 4 for 128³ pipeline
    
    for z in reversed(range(vis_res)):
        row_chars = ""
        for x in range(vis_res):
            # Inspect pooled spatial blocks to consolidate multi-resolution grids safely
            block = slice_side[z*scale : (z+1)*scale, x*scale : (x+1)*scale]
            row_chars += "■" if np.any(block > 0) else "  "
            
        if "■" in row_chars:
            print(f"Z={z:02d} | {row_chars}")
            
    # Append the resolution tag to file name to isolate cache benchmarks safely
    output_npy = f"data/processed_voxels/test_sphere_{resolution}res.npy"
    np.save(output_npy, voxel_matrix)
    print(f"\n5. Voxel data matrix exported successfully: {output_npy}")

if __name__ == "__main__":
    os.makedirs("data/raw_stl", exist_ok=True)
    os.makedirs("data/processed_voxels", exist_ok=True)
    
    test_pipeline()