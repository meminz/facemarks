import numpy as np
import open3d as o3d

from ._mp_utils import _mp_image
from ._geometry_processing import _hpr_mesh_based, _perspective_rays_directions
from ._triangles import TRIANGLES

NUM_FACEMARKS = 478
DEFAULT_FACEMARKS = 468  # MediaPipe's photo-only iris landmarks (468-477) are excluded by default, but can be requested explicitly
IMG_SIZE = 720


def _sample_camera_rotations(projections):
    y_rots = np.random.uniform(-np.pi/4, np.pi/4, 	projections)
    x_rots = np.random.uniform(0,		 np.pi/8, 	projections)

    return [ np.asarray(o3d.geometry.get_rotation_matrix_from_axis_angle([x,y,0])) for x,y in zip(x_rots, y_rots) ]


def _setup_offscreen_viewer(textured_mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=IMG_SIZE, height=IMG_SIZE)
    vis.get_render_option().background_color = [0,0,0]
    vis.add_geometry(textured_mesh)

    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=-10)

    vis.update_renderer()

    intr_mat = ctr.convert_to_pinhole_camera_parameters().intrinsic.intrinsic_matrix

    return vis, ctr, intr_mat


def _hidden_point_removal(detection_result, landmark_ids):
    mp_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor([[p.x,-p.y,-p.z] for p in detection_result.face_landmarks[0]], dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(TRIANGLES)
    )
    mp_mesh.translate( - mp_mesh.get_axis_aligned_bounding_box().get_center().numpy()  )

    visible_points = _hpr_mesh_based(mp_mesh, [0,0,1])
    visible_points = visible_points[np.isin(visible_points, landmark_ids)]

    landmarks = [ [p.x,p.y,0] for p in detection_result.face_landmarks[0] ]
    landmarks_2d = np.asarray(landmarks)[visible_points]

    return visible_points, landmarks_2d


def _world_rays_from_camera(camera_r, landmarks_2d, intr_mat, extr_mat):
    persp_rays = _perspective_rays_directions(landmarks_2d, IMG_SIZE, intr_mat)

    world_rays = (persp_rays * [1,-1,-1]) @ np.linalg.inv(camera_r)

    camera_pos = camera_r @ (np.asarray([0,0,1]) * extr_mat[2,3])

    return camera_pos, world_rays


def _project_views(detector, textured_mesh, camera_rots, landmark_ids):
    views = {i:[] for i in landmark_ids}
    successful_detections = 0

    vis, ctr, intr_mat = _setup_offscreen_viewer(textured_mesh)

    for camera_r in camera_rots:
        ctr.set_front(camera_r @ [0,0,1])
        ctr.set_lookat([0,0,0])
        vis.update_renderer()
        
        img = (np.asarray(vis.capture_screen_float_buffer(True)) * 255 ).astype(np.uint8)

        detection_result = detector.detect(_mp_image(img))
        if not detection_result.face_landmarks: continue

        successful_detections += 1

        # HPR
        visible_points, landmarks_2d = _hidden_point_removal(detection_result, landmark_ids)

        vis.update_renderer()
        extr_mat = ctr.convert_to_pinhole_camera_parameters().extrinsic

        camera_pos, world_rays = _world_rays_from_camera(camera_r, landmarks_2d, intr_mat, extr_mat)

        for i,r in zip(visible_points, world_rays):
            views[i].append(
                np.asarray(
                    [*camera_pos, *r],
                    dtype=np.float32
                )
            )

    return views, successful_detections
