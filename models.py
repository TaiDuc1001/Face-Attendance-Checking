from deepface import DeepFace
from facenet_pytorch import MTCNN, InceptionResnetV1
from config import *

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
        "gamma": 0.2,
    },
    "ArcFace": {
        "model": ArcFaceModel().model,
        "data_path": ARCFACE_DATA_PATH,
        "alpha": 1,
        "gamma": 0.25,
    },
    "VGG-Face": {
        "model": VGGFaceModel().model,
        "data_path": VGGFACE_DATA_PATH,
        "alpha": 1,
        "gamma": 0.55,
    }
}
