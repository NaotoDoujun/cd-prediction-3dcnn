import numpy as np

def show_pointcloud_with_open3d(points):
    """
    Accepts a sampled pointcloud array and launches an interactive Open3D viewer window.
    """
    import open3d as o3d  # Lazy load to keep core dependencies lightweight
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.0, 0.8, 1.0]) # Cyan layout
    
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    
    print("\n--- PointNet++ Input Point Cloud Viewer Initialized ---")
    print(f"  - Total Sampled Vehicle Nodes (Points): {len(points)}")
    print("Close the visualizer window to resume script execution.\n")
    
    o3d.visualization.draw_geometries([pcd, axes], 
                                      window_name="PointNet++ Normalized Point Cloud Preview",
                                      width=1152, height=864)

def show_pointnet_overlay_with_open3d(stl_path, sampled_points, num_points):
    """
    Renders an interactive Open3D overlay preview containing both the 
    sampled cyan point cloud and the structurally synchronized crimson CAD mesh.
    """
    import open3d as o3d
    import trimesh
    from src.preprocess_pointnet import get_mesh_alignment_params

    print("\nTriggering Point Cloud + CAD Mesh Fusion Overlay Preview...")
    
    # A. Configure sampled point cloud geometry (Cyan / Blue nodes)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled_points)
    pcd.paint_uniform_color([0.0, 0.8, 1.0]) 
    
    # B. Load and align the original CAD mesh using unified preprocess definitions
    raw_mesh = trimesh.load(stl_path)
    translation, max_extent = get_mesh_alignment_params(raw_mesh)
    
    orig_mesh = o3d.io.read_triangle_mesh(stl_path)
    orig_mesh.compute_vertex_normals()
    
    # Synchronize translation & scale transformations perfectly
    orig_mesh.translate(translation)
    if max_extent > 0:
        orig_mesh.scale(1.0 / max_extent, center=np.array([0, 0, 0]))
    
    # Paint the CAD mesh translucent Crimson Red
    orig_mesh.paint_uniform_color([0.8, 0.2, 0.2])
    
    # C. Build and activate Custom Open3D Visualizer Window
    vis = o3d.visualization.Visualizer()
    window_title = f"PointNet++ Inference Preview ({num_points} pts Overlay)"
    vis.create_window(window_name=window_title, width=1152, height=864)
    
    vis.add_geometry(pcd)
    vis.add_geometry(orig_mesh)
    
    # Local coordinate origin reference axes frame
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    vis.add_geometry(axes)
    
    # Apply modern structural wireframe rendering specs
    render_option = vis.get_render_option()
    render_option.mesh_show_wireframe = True
    render_option.point_size = 4.0
    
    print("\n--- Interactive Point-Mesh Fusion Viewer Initialized ---")
    print("  - Cyan Nodes (Points) : Sampled PointNet++ Input Structure")
    print("  - Red Wireframe Mesh  : Aligned Original CAD Silhouette Reference")
    print("Close the visualizer window to resume script execution.\n")
    
    vis.run()
    vis.destroy_window()