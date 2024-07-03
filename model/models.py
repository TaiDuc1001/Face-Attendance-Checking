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

class Facenet512:
    def __init__(self):
        self.model = DeepFace.represent

alpha = 1
# Models dict
model_dict = {
    "resnet": {
        "model": ResnetModel().model,
        "data_path": RESNET_DATA_PATH,
        "alpha": alpha,
        "gamma": 0.7,
    },
    "Facenet512": {
        "model": Facenet512().model,
        "data_path": FACENET512_DATA_PATH,
        "alpha": alpha,
        "gamma": 0.2,
    },
    "VGG-Face": {
        "model": VGGFaceModel().model,
        "data_path": VGGFACE_DATA_PATH,
        "alpha": alpha,
        "gamma": 0.05,
    }
}
