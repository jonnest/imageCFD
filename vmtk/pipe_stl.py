import meshio
import trimesh
import numpy as np
from stl import mesh
import numpy as np
import pyvista as pv

path = f'/Users/jonathan/Documents/applications/superres/ubend/'
target = path + 'cylinder_with_patches.stl'

cyl = pv.Cylinder()

p = pv.Plotter(shape=(1, 1))
# Top row
p.subplot(0, 0)
p.add_mesh(cyl, color='lightblue', show_edges=True)
p.show()
1/0

def create_cylinder_with_patches(height=10, radius=5, radial_segments=32):
    # Create a cylindrical mesh
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=radial_segments)
    
    # Split faces based on z-coordinates to define boundaries
    wall_faces = []
    boundary_0_faces = []
    boundary_1_faces = []
    
    for i, face in enumerate(cylinder.faces):
        # Identify each face type by z-coordinate
        z_coords = cylinder.vertices[face][:, 2]
        if np.allclose(z_coords, height / 2):
            boundary_1_faces.append(face)
        elif np.allclose(z_coords, -height / 2):
            boundary_0_faces.append(face)
        else:
            wall_faces.append(face)
    
    # Prepare ASCII STL with patches as comments
    stl_data = []

    # Function to convert a face to ASCII STL format
    def face_to_stl(face):
        vertices = cylinder.vertices[face]
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        normal /= np.linalg.norm(normal)  # Normalize
        lines = [
            f"  facet normal {normal[0]} {normal[1]} {normal[2]}",
            "    outer loop"
        ]
        for vertex in vertices:
            lines.append(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}")
        lines.append("    endloop")
        lines.append("  endfacet")
        return "\n".join(lines)
    
    # Append faces with comments as patches
    stl_data.append("solid cylinder_with_patches")
    stl_data.append("  // wall patch")
    for face in wall_faces:
        stl_data.append(face_to_stl(face))
        
    stl_data.append("  // boundary_0 patch")
    for face in boundary_0_faces:
        stl_data.append(face_to_stl(face))
        
    stl_data.append("  // boundary_1 patch")
    for face in boundary_1_faces:
        stl_data.append(face_to_stl(face))
        
    stl_data.append("endsolid cylinder_with_patches")
    
    # Write to an ASCII STL file
    with open(path + 'cylinder_with_patches.stl', 'w') as file:
        file.write("\n".join(stl_data))
    
    print("ASCII STL file created with patch names in comments: 'cylinder_with_patches.stl'")

# Run the function to create and save the STL file
create_cylinder_with_patches()