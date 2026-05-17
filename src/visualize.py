import numpy as np
import open3d as o3d

def show_voxels_with_open3d(voxel_matrix):
    """
    Accepts a generated voxel matrix (0.0 or 1.0) and visualizes it using Open3D.
    Can be imported and reused across any script in the pipeline.
    """
    # 1. Extract indices where voxels are active (> 0)
    indices = np.argwhere(voxel_matrix > 0)
    
    if len(indices) == 0:
        print("Warning: Voxel matrix is empty (no active geometry detected).")
        return

    # 2. Create an Open3D PointCloud object from indices
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(indices)
    
    # 3. Apply custom coloring for better geometric depth visibility (Slate Blue)
    color_array = np.tile([0.3, 0.5, 0.7], (len(indices), 1))
    pcd.colors = o3d.utility.Vector3dVector(color_array)
    
    # 4. Generate the VoxelGrid layout with unit size (pitch=1.0 alignment)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
    
    # 5. Add origin coordinate axes to verify spatial alignment and ground clearance
    # Fixed for Open3D >= 0.18.0: Changed create_coordinate_axis to create_coordinate_frame
    # Red: X-axis, Green: Y-axis, Blue: Z-axis
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=voxel_matrix.shape[0] * 0.2)
    
    # 6. Launch the Open3D visualization window
    print("\n--- 3D Voxel Viewer Initialized ---")
    print("  - Left Drag       : Rotate View")
    print("  - Right Drag/Wheel: Zoom In/Out")
    print("  - Shift + Left Drag: Pan View")
    print("Close the visualizer window to resume script execution.\n")
    
    o3d.visualization.draw_geometries([voxel_grid, axes], 
                                      window_name="3D CNN Input Geometry Preview",
                                      width=1024, height=768)