import numpy as np
import trimesh

def get_mesh_alignment_params(mesh):
    """
    Computes exact geometric alignment parameters from a trimesh object.
    Ensures centering via mass centroid and locking bottom floor to Z=0.
    """
    centroid = mesh.centroid
    bounds = mesh.bounds
    max_extent = np.max(mesh.extents)
    
    # Translation vector: Center X-Y mass, drop bottom Z floor to 0
    translation = np.array([-centroid[0], -centroid[1], -bounds[0][2]])
    
    return translation, max_extent

def convert_stl_to_aligned_pointcloud(stl_path, num_points=2048):
    """
    Loads an STL mesh, uniformly samples N points from its surface,
    and applies standard aerodynamic coordinate normalization.
    """
    mesh = trimesh.load(stl_path)
    
    # Uniform surface sampling
    points = mesh.sample(num_points)
    points = np.array(points, dtype=np.float32)
    
    # Fetch unified transformation parameters
    translation, max_extent = get_mesh_alignment_params(mesh)
    
    # Apply translation transformation
    points[:, 0] += translation[0]
    points[:, 1] += translation[1]
    points[:, 2] += translation[2]
    
    # Apply scale normalization context
    if max_extent > 0:
        points /= max_extent
        
    return points