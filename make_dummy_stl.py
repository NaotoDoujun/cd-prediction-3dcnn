import os
import trimesh

def create_dummy_stl():
    # Define paths inside the standard data directory structure
    target_dir = "data/raw_stl"
    os.makedirs(target_dir, exist_ok=True)
    
    # Create a simple box geometry representing a primitive vehicle body bounds
    # dimensions: length=4.5m, width=1.8m, height=1.4m
    mesh = trimesh.creation.box(extents=[4.5, 1.8, 1.4])
    
    # Save to the designated raw data path
    output_path = os.path.join(target_dir, "new_design_test.stl")
    mesh.export(output_path)
    print(f"Successfully generated dummy STL file at: {output_path}")

if __name__ == "__main__":
    create_dummy_stl()