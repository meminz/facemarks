import open3d as o3d
import operator

from ._mp_utils import _detectorInit
from ._projection import _sample_camera_rotations, _project_views, NUM_FACEMARKS, DEFAULT_FACEMARKS
from ._geometry_processing import _hit_coords, _consensus_point, _closest_vertex_ids



def predict(meshes, projections=100, landmarks=tuple(range(DEFAULT_FACEMARKS))):
    actual_mesh = meshes["original"]
    textured_mesh = meshes["textured"]
    mesh_t = meshes["tensor"]

    landmark_ids = _validate_landmark_ids(landmarks)


### PROJECTIONS AND LANDMARKS
    detector = _detectorInit()
    camera_rots = _sample_camera_rotations(projections)
    views, successful_detections = _project_views(detector, textured_mesh, camera_rots, landmark_ids)

    if successful_detections == 0: print(f"Error detecting face."); return


### RAYCASTING
    landmarks_3d = _reconstruct_landmarks(mesh_t, views)
    closest_vertex_ids = _map_to_closest_vertices(actual_mesh.vertices, landmarks_3d)


    return {
        "facemarks_3d": landmarks_3d,
        "closest_vertex_ids": closest_vertex_ids
    }


def _validate_landmark_ids(landmarks):
    try:
        landmark_ids = tuple(landmarks)
    except TypeError:
        raise ValueError("landmarks must be an iterable of landmark indices") from None

    try:
        landmark_ids = tuple(operator.index(i) for i in landmark_ids)
    except TypeError:
        raise ValueError("landmark indices must be integers") from None

    if not landmark_ids:
        raise ValueError("landmarks must not be empty")

    if len(set(landmark_ids)) != len(landmark_ids):
        raise ValueError("landmark indices must be unique")

    if any(i < 0 or i >= NUM_FACEMARKS for i in landmark_ids):
        raise ValueError(f"landmark indices must be in 0..{NUM_FACEMARKS - 1}")

    return landmark_ids


def _reconstruct_landmarks(mesh_t, views):
    positions = [None] * (max(views) + 1)

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

        positions[i] = _consensus_point(hits).tolist()

    return positions


def _map_to_closest_vertices(vertices, landmarks_3d):
    closest_vertex_ids = [None] * len(landmarks_3d)

    resolved = [k for k, coords in enumerate(landmarks_3d) if coords is not None]
    if resolved:
        for k, vertex_id in zip(resolved, _closest_vertex_ids(vertices, [landmarks_3d[k] for k in resolved])):
            closest_vertex_ids[k] = vertex_id

    return closest_vertex_ids