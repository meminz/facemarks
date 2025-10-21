import open3d as o3d
import numpy as np
import scipy

from ._mp_utils import _detectorInit, _mpImage
from ._geometry_processing import _hpr_mesh_based, _perspective_rays_directions, _hit_coords

from ._triangles import TRIANGLES
IMG_SIZE = 720



def predict(meshes, projections=100):
    actual_mesh, textured_mesh, mesh_t = meshes.values()


### PROJECTIONS AND LANDMARKS
    detector = _detectorInit()

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=IMG_SIZE, height=IMG_SIZE)
    vis.get_render_option().background_color = [0,0,0]
    vis.add_geometry(textured_mesh)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=-10)

    vis.update_renderer()

    intr_mat = ctr.convert_to_pinhole_camera_parameters().intrinsic.intrinsic_matrix
    extr_mat = ctr.convert_to_pinhole_camera_parameters().extrinsic


    y_rots = np.random.uniform(-np.pi/4, np.pi/4, 	projections)
    x_rots = np.random.uniform(0,		 np.pi/8, 	projections)
    camera_rots = [ np.asarray(o3d.geometry.get_rotation_matrix_from_axis_angle([x,y,0])) for x,y in zip(x_rots, y_rots) ]


    views = {i:[] for i in range(478)}

    for camera_r in camera_rots:
        ctr.set_front(camera_r @ [0,0,1])
        ctr.set_lookat([0,0,0])
        vis.update_renderer()
        
        img = (np.asarray(vis.capture_screen_float_buffer(True)) * 255 ).astype(np.uint8)

        detection_result = detector.detect(_mpImage(img))
        if not detection_result.face_landmarks: continue

    ### HPR
        mp_mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor([[p.x,-p.y,-p.z] for p in detection_result.face_landmarks[0]], dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(TRIANGLES)
        )
        mp_mesh.translate( - mp_mesh.get_axis_aligned_bounding_box().get_center().numpy()  )

        visible_points = _hpr_mesh_based(mp_mesh, [0,0,1])

        landmarks = [ [p.x,p.y,0] for p in detection_result.face_landmarks[0] ]
        landmarks = np.asarray(landmarks)[visible_points]

        persp_rays = _perspective_rays_directions(landmarks, IMG_SIZE, intr_mat)

        world_rays = (persp_rays * [1,-1,-1]) @ np.linalg.inv(camera_r)

        vis.update_renderer()
        extr_mat = ctr.convert_to_pinhole_camera_parameters().extrinsic


        camera_pos_cam = np.array([0, 0, 0, 1])
        camera_pos_world = np.linalg.inv(extr_mat) @ camera_pos_cam
        camera_pos = camera_pos_world[:3]

        camera_pos = camera_r @ (np.asarray([0,0,1]) * extr_mat[2,3])

        for i,r in zip(visible_points, world_rays):
            views[i].append(
                np.asarray(
                    [*camera_pos, *r],
                    dtype=np.float32
                )
            )

    if len(views) < projections/2: print(f"Error detecting face."); return


### RAYCASTING
    landmarks_3d = []

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    for i,rays in views.items():

        if len(rays)==0: print(f"No rays for landmark {i}."); continue
        ans = scene.cast_rays(rays)
        hits = _hit_coords(ans,rays)
        if len(hits)==0: print(f"No hits for landmark {i}."); continue

        distances = scipy.spatial.distance.cdist(hits, hits)
        means = [np.square(np.mean(x)) for x in distances]

        landmarks_3d.append( hits[np.argmin(means)].tolist() )


    vertices_distances = scipy.spatial.distance.cdist(np.asarray(landmarks_3d), np.asarray(actual_mesh.vertices))
    closest_vertex_ids = [ int(np.argmin(x)) for x in vertices_distances ]


    return {
        "facemarks_3d": landmarks_3d,
        "closest_vertex_ids": closest_vertex_ids
    }
