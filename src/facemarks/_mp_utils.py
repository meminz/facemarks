import numpy as np
import open3d as o3d
import mediapipe as mp
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def _mpImage(img):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=img)


def _detectorInit(detection_confidence=.5):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions( model_asset_path= "./face_landmarker.task" ),
            min_face_detection_confidence = detection_confidence,
            running_mode = vision.RunningMode.IMAGE,

            output_face_blendshapes = False,
            output_facial_transformation_matrixes = False,
        )

    return vision.FaceLandmarker.create_from_options(options)
