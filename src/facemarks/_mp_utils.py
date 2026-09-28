import os
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

def _mp_image(img):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=img)


def _detectorInit(detection_confidence=.5):
    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(
            f"MediaPipe Face Landmarker model not found at {os.path.abspath(MODEL_FILENAME)}.\n"
            f"predict() needs this file in the current working directory. Download it with:\n"
            f"    wget -O {MODEL_FILENAME} {MODEL_URL}"
        )

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions( model_asset_path= os.path.abspath(MODEL_FILENAME) ),
            min_face_detection_confidence = detection_confidence,
            running_mode = vision.RunningMode.IMAGE,

            output_face_blendshapes = False,
            output_facial_transformation_matrixes = False,
        )

    return vision.FaceLandmarker.create_from_options(options)
