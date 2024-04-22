from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn.functional as F
from PIL import Image
from helper import timing
from config import DATA_PATH

# Init mtcnn and resnet model
mtcnn = MTCNN(image_size=216)
resnet = InceptionResnetV1(pretrained="vggface2").eval()

@timing
def face_match(image_path, data_path, return_difference_list=False):
    img = Image.open(image_path)
    face = mtcnn(img)
    embeddings = resnet(face.unsqueeze(0)).detach()
    saved_data = torch.load(data_path)
    embedding_list = saved_data[0]
    name_list = saved_data[1]
    difference_list = []
    for old_embedding in embedding_list:
        similarity = F.cosine_similarity(embeddings, old_embedding).item()
        difference_list.append(similarity)
        
    idx_min = difference_list.index(max(difference_list))

    if return_difference_list:
        return name_list[idx_min], difference_list[idx_min], difference_list
    else:
        return name_list[idx_min], difference_list[idx_min]
    
person, difference, difference_list = face_match(
    image_path="captured_image.jpg", 
    data_path=DATA_PATH, 
    return_difference_list=True
)

print(person)
print(difference)
print(difference_list)