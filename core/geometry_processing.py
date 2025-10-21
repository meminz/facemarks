import open3d as o3d
import numpy as np


def _meshes_setup(meshes, offset=[0,0,0], rotation=None):
    actual_mesh, textured_mesh, _ = meshes.values()

    scale = max(textured_mesh.get_max_bound() - textured_mesh.get_min_bound())
    textured_mesh.scale(1/scale, textured_mesh.get_axis_aligned_bounding_box().get_center())
    if rotation is not None:
        textured_mesh.rotate(rotation[0])
    textured_mesh.translate( - textured_mesh.get_axis_aligned_bounding_box().get_center() - offset )

    actual_mesh.scale(1/scale, actual_mesh.get_axis_aligned_bounding_box().get_center())
    if rotation is not None:
        actual_mesh.rotate(rotation[0])
    actual_mesh.translate( - actual_mesh.get_axis_aligned_bounding_box().get_center() - offset )

    mesh_t = o3d.t.geometry.TriangleMesh(
        device=o3d.core.Device("CUDA:1" if o3d.core.cuda.is_available() else "CPU:0")
    ).from_legacy(textured_mesh)

    meshes.update({"tensor": mesh_t})

    return meshes


def _hit_coords(ans, np_rays):
	hit_coords = []

	for i,r in enumerate(np_rays):
		distance = ans["t_hit"][i].numpy()
		if distance == float("inf"): continue

		hit_coords.append(r[:3] + r[3:] * distance)

	return np.asarray(hit_coords)



def _hpr_mesh_based(mesh: o3d.t.geometry.TriangleMesh, eye=[0,0,0]):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    pcd_points = mesh.vertex.positions.numpy()
    distance_vectors = [point-eye for point in pcd_points]
    rays = np.asarray([[*eye, *direction] for direction in map(lambda vec: vec/np.linalg.norm(vec), distance_vectors)]).astype(np.float32)
    cast_results = scene.cast_rays(rays)

    hit_distances = cast_results["t_hit"].numpy()
    visibility_mask = (hit_distances + 0.00001) >= np.linalg.norm(distance_vectors, axis=1)

    return np.asarray(visibility_mask).nonzero()[0]


def _perspective_rays_directions(img_landmarks, size, intrinsic):
    return np.asarray( list(
        map(lambda x:x/np.linalg.norm(x),
            [np.linalg.inv(intrinsic) @ np.asarray([p[0]*size, p[1]*size, 1])
             for p in img_landmarks]
            )
        ) )

def _align_vector_to_xz(eyes_landmarks):
    eyes_vec = eyes_landmarks[1] - eyes_landmarks[0]
    eyes_vec /= np.linalg.norm(eyes_vec)

    angle = np.arctan2(eyes_vec[1], eyes_vec[0])
    return np.asarray( o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,-angle]) )


def _align_vector_to_xaxis(eyes_landmarks):
    eyes_vec = eyes_landmarks[1] - eyes_landmarks[0]
    eyes_vec /= np.linalg.norm(eyes_vec)

    Xaxis = np.array([1,0,0])

    rotation_axis = np.dot(eyes_vec, Xaxis)
    axis_norm = np.linalg.norm(rotation_axis)

    if np.linalg.norm(rotation_axis) <= 1e-8:
        return np.eye(3)

    rotation_axis /= axis_norm

    angle = np.arccos( np.clip( np.dot(eyes_vec, Xaxis), -1, 1  ) )

    return np.asarray( o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,-angle]) )
