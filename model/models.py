import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

from deepface import DeepFace
from facenet_pytorch import MTCNN, InceptionResnetV1
from scripts.config import *

mtcnn = MTCNN()
class ResnetModel:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained="vggface2").eval()

class VGGFaceModel:
    def __init__(self):
        self.model = DeepFace.represent

class ArcFaceModel:
    def __init__(self):
        self.model = DeepFace.represent

# Models dict
model_dict = {
    "resnet": {
        "model": ResnetModel().model,
        "data_path": RESNET_DATA_PATH,
        "alpha": 1,
        "gamma": 0.35,
    },
    "ArcFace": {
        "model": ArcFaceModel().model,
        "data_path": ARCFACE_DATA_PATH,
        "alpha": 0.9,
        "gamma": 0.1,
    },
    "VGG-Face": {
        "model": VGGFaceModel().model,
        "data_path": VGGFACE_DATA_PATH,
        "alpha": 0.8,
        "gamma": 0.55,
    }
}
