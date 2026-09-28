import open3d as o3d

from ._mp_utils import _detectorInit
from ._projection import _sample_camera_rotations, _project_views
from ._geometry_processing import _hit_coords, _consensus_point, _closest_vertex_ids



def predict(meshes, projections=100):
    actual_mesh = meshes["original"]
    textured_mesh = meshes["textured"]
    mesh_t = meshes["tensor"]


### PROJECTIONS AND LANDMARKS
    detector = _detectorInit()
    camera_rots = _sample_camera_rotations(projections)
    views, successful_detections = _project_views(detector, textured_mesh, camera_rots)

    if successful_detections == 0: print(f"Error detecting face."); return


### RAYCASTING
    landmarks_3d = _reconstruct_landmarks(mesh_t, views)
    closest_vertex_ids = _map_to_closest_vertices(actual_mesh.vertices, landmarks_3d)


    return {
        "facemarks_3d": landmarks_3d,
        "closest_vertex_ids": closest_vertex_ids
    }


def _reconstruct_landmarks(mesh_t, views):
    landmarks = [None] * len(views)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    for i,rays in views.items():

        if len(rays)==0:
            print(f"No rays for landmark {i}.")
            continue

        ans = scene.cast_rays(rays)
        hits = _hit_coords(ans,rays)
        if len(hits)==0:
            print(f"No hits for landmark {i}.")
            continue

        landmarks[i] = _consensus_point(hits).tolist()

    assert len(landmarks) == len(views), "facemark slots lost during raycasting."

    return landmarks


def _map_to_closest_vertices(vertices, landmarks_3d):
    closest_vertex_ids = [None] * len(landmarks_3d)

    resolved = [k for k, coords in enumerate(landmarks_3d) if coords is not None]
    if resolved:
        for k, vertex_id in zip(resolved, _closest_vertex_ids(vertices, [landmarks_3d[k] for k in resolved])):
            closest_vertex_ids[k] = vertex_id

    return closest_vertex_ids