import open3d as o3d
import json
from ._geometry_processing import _meshes_setup

def import_mesh_and_setup(filename):
    # assert os.path.exists(mesh_path), "The mesh\'s filepath should be a valid path."
    # assert mesh_path.lower().endswith(".obj"), "Only .obj mesh files are supported. sry :("

    try: textured_mesh = o3d.io.read_triangle_mesh(filename, True)
    except: print("Error importing mesh.\n"); return

    if not textured_mesh.vertices: print(f"Error loading mesh.\n"); return

    actual_mesh = o3d.io.read_triangle_mesh(filename)

    return _meshes_setup({
        "original": actual_mesh,
        "textured": textured_mesh,
        "tensor": None
    })


def save_facemarks_json(input_path, prediction_result, json_path):
    data = {
        "model": file,
        "normalized coordinates": prediction_result["landmarks_3d"],
        "closest vertex indexes": prediction_result["closest_vertices_ids"]
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
