import os
import numpy as np
import trimesh
from src.preprocess import convert_stl_to_aligned_voxels

def test_pipeline():
    print("1. Generating a local 3D mesh (Sphere) for verification...")
    # Generate a standard sphere with a radius of 0.5 using trimesh built-in functions
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    
    # Shift geometry away from origin to simulate poor CAD exports and test rigid alignment
    mesh.apply_translation([10.0, -5.0, 8.5])
    
    test_stl_path = "data/raw_stl/test_sphere.stl"
    mesh.export(test_stl_path)
    print(f"   -> Saved misaligned test STL file to: {test_stl_path}")
    
    print("\n2. Executing voxelization pipeline with rigid alignment (Resolution: 32)...")
    resolution = 32
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
    
    # Y-axis (Lateral): Check if center of mass maps near grid midpoint (32 / 2 = 15.5)
    mean_y = np.mean(y_indices)
    print(f"   [Y-axis (Lateral)] Center of mass index: {mean_y:.2f} (Expected: ~15.5 -> Central symmetry check)")
    
    # X-axis (Longitudinal): Check if center of mass maps near grid midpoint
    mean_x = np.mean(x_indices)
    print(f"   [X-axis (Longitudinal)] Center of mass index: {mean_x:.2f} (Expected: ~15.5 -> Central position check)")

    # ---------------------------------------------------------
    # 4. Terminal Voxel Visualizer (Cross-Section Slice)
    # ---------------------------------------------------------
    print("\n4. Displaying mid-plane cross-section (Side view profile):")
    mid_y = resolution // 2
    slice_side = voxel_matrix[:, mid_y, :]  # 2D cross-section array [Z, X]
    
    for z in reversed(range(resolution)):
        row_chars = "".join(["■" if slice_side[z, x] > 0 else "  " for x in range(resolution)])
        if "■" in row_chars:
            print(f"Z={z:02d} | {row_chars}")
            
    # Save test output
    output_npy = "data/processed_voxels/test_sphere.npy"
    np.save(output_npy, voxel_matrix)
    print(f"\n5. Voxel data matrix exported successfully: {output_npy}")

if __name__ == "__main__":
    os.makedirs("data/raw_stl", exist_ok=True)
    os.makedirs("data/processed_voxels", exist_ok=True)
    
    test_pipeline()