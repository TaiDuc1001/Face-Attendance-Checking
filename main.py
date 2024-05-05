import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.models import vgg16
import torch.nn.functional as F

from PIL import Image
from helper import timing
import os
from config import RESNET_DATA_PATH, VGG_DATA_PATH, EXTRACTED_FACES_PATH, THRESHOLD, VGG_WEIGHTS_PATH

# Init mtcnn, resnet and vgg16 model
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained="vggface2").eval()
vgg_model = vgg16(pretrained=False)
vgg_model.load_state_dict(torch.load(VGG_WEIGHTS_PATH))
vgg_model.eval()

def calculate_score(predictions, targets, alpha):
    cosine_similarity = F.cosine_similarity(predictions, targets).item()
    l2_distance = F.pairwise_distance(predictions, targets).item()
    score = alpha * cosine_similarity + (1 - alpha) * l2_distance
    return score

@timing
def face_match(image_path, data_path, model):
    img = Image.open(image_path)
    face = mtcnn(img)
    model = resnet if model == "resnet" else vgg_model

    if face is not None:
        face = face.unsqueeze(0)
        embeddings = model(face).detach()

        # === Load data ===
        saved_data = torch.load(data_path)
        embedding_list = saved_data[0]
        name_list = saved_data[1]

        score_list = []
        for old_embeddings in embedding_list:
            score = calculate_score(embeddings, old_embeddings, alpha=0.8)
            score_list.append(score)
            
        idx_min = score_list.index(max(score_list)) if max(score_list) >= THRESHOLD else None

        if idx_min is not None:
            return name_list[idx_min], score_list[idx_min], score_list
        else:
            return None, None, None
    else:
        return None, None, None


image_files = os.listdir(EXTRACTED_FACES_PATH)
for image_file in image_files:
    image_path = os.path.join(EXTRACTED_FACES_PATH, image_file)
    r_person, r_score, _ = face_match(image_path=image_path, data_path=RESNET_DATA_PATH, model="resnet")
    v_person, v_score, _ = face_match(image_path=image_path, data_path=VGG_DATA_PATH, model="vgg")
    if r_person is not None or v_person is not None:
        print(f"{image_file} --- R_Person: {r_person} --- V_Person: {v_person} --- R_Similarity: {r_score:.4f} --- V_Similarity: {v_score:.4f}\n")
    else:
        print(f"{image_file}: no face detected.\n")
    