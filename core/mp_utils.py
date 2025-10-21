import numpy as np
import open3d as o3d
import mediapipe as mp
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def mpImage(img):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=img)


def detectorInit(detection_confidence=.5):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions( model_asset_path= "./face_landmarker_v2.task" ),
            min_face_detection_confidence = detection_confidence,
            running_mode = vision.RunningMode.IMAGE,

            output_face_blendshapes = False,
            output_facial_transformation_matrixes = False,
        )

    return vision.FaceLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

    # FACE MESH
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )

    # CONTOURS
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_contours_style()
        )
    
    # EYES
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    return annotated_image
