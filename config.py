import logging
from pathlib import Path
import sys

import cv2 as cv

from face_detect import FaceDetect


logging.basicConfig(level=logging.DEBUG, 
                        stream=sys.stdout)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / 'intel/face-detection-retail-0005/FP16/face-detection-retail-0005'
CAMERA_PORT = 0
logger.debug({'PROJECT_ROOT': PROJECT_ROOT,
              'MODEL_PATH': MODEL_PATH,
              'CAMERA_PORT': CAMERA_PORT})

capture = cv.VideoCapture(CAMERA_PORT)
face_detect = FaceDetect(model_path=str(MODEL_PATH))
