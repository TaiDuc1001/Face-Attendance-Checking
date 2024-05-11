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

# Models dict
model_dict = {
    "resnet": {
        "model": ResnetModel().model,
        "data_path": RESNET_DATA_PATH,
        "alpha": 0.8,
        "gamma": 0.8,
    },
    "vgg-face": {
        "model": VGGFaceModel().model,
        "data_path": VGG_FACE_DATA_PATH,
        "alpha": 0.6,
        "gamma": 0.2,
    }
}
