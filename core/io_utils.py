import open3d as o3d
import json

def import_mesh(filename):
    # assert os.path.exists(mesh_path), "The mesh\'s filepath should be a valid path."
    # assert mesh_path.lower().endswith(".obj"), "Only .obj mesh files are supported. sry :("

    try: textured_mesh = o3d.io.read_triangle_mesh(filename, True)

    except: print("Error importing mesh.\n"); return

    if not textured_mesh.vertices: print(f"Error loading mesh."); return

    actual_mesh = o3d.io.read_triangle_mesh(filename)

    return {
        "original": actual_mesh,
        "textured": textured_mesh,
        "tensor": None,
    }


def save_facemarks_json(input_path, landmarks_3d, closest_vertices_ids, json_path):
    data = {
        "model": file,
        "normalized coordinates": landmarks_3d,
        "closest vertex indexes": closest_vertices_ids
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
