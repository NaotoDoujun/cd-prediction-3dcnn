import os
import trimesh

def create_perfect_capsule_stl():
    """
    Generates a high-fidelity built-in capsule mesh (pill shape) from trimesh.
    This guarantees 100% offline reliability without relying on remote URLs,
    and yields a smooth, aerodynamic geometry perfect for pipeline debugging.
    """
    print("Generating high-density geometric capsule sample...")

    # Create a smooth capsule (length=4.5m, radius=0.7m)
    # This represents a generic aerodynamic bullet/capsule train profile
    mesh = trimesh.creation.capsule(height=3.1, radius=0.7)
    
    # Rotate it 90 degrees around Y axis so it lies horizontally along the X axis (like a vehicle)
    # Trimesh applies a standard transformation matrix
    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        angle=1.5708, # 90 degrees in radians
        direction=[0, 1, 0]
    ))

    # Define paths inside the standard data directory structure
    target_dir = "data/raw_stl"
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, "new_design_test.stl")
    
    # Export the high-fidelity geometry as an STL binary
    mesh.export(output_path)
    print(f"Successfully generated a beautiful sample STL (Capsule) at: {output_path}")

if __name__ == "__main__":
    create_perfect_capsule_stl()