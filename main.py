from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import time

def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Executed for {func.__name__}: {end_time-start_time} seconds.")
        return result
    return wrapper

mtcnn = MTCNN(image_size=216)
resnet = InceptionResnetV1(pretrained="vggface2").eval()


dataset = datasets.ImageFolder("fake_dataset")
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

def collate_func(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_func)

@timing
def get_embed_data(loader):
    embeddings_dict = {}
    for img, idx in loader:
        face, prob = mtcnn(img, return_prob=True)
        if face is not None and prob >= 0.9:
            embeddings = resnet(face.unsqueeze(0)).detach()
            if idx not in embeddings_dict:
                embeddings_dict[idx] = [embeddings]
            else:
                embeddings_dict[idx].append(embeddings)

    aggregated_embeddings = []
    name_list = []
    for idx, embeddings_list in embeddings_dict.items():
        avg_embedding = torch.stack(embeddings_list).mean(dim=0)
        aggregated_embeddings.append(avg_embedding)
        name_list.append(idx_to_class[idx])

    data = [aggregated_embeddings, name_list]
    torch.save(data, "data.pt")

@timing
def face_match(image_path, data_path):
    img = Image.open(image_path)
    face, prob = mtcnn(img, return_prob=True)
    embeddings = resnet(face.unsqueeze(0)).detach()
    saved_data = torch.load(data_path)
    embedding_list = saved_data[0]
    name_list = saved_data[1]
    difference_list = []
    for old_embedding in embedding_list:
        # distance = torch.dist(embeddings, old_embedding).item()
        similarity = F.cosine_similarity(embeddings, old_embedding).item()
        difference_list.append(similarity)
        
    idx_min = difference_list.index(max(difference_list))
    return name_list[idx_min], difference_list[idx_min]

# get_embed_data(loader=loader)
person, difference_list = face_match(image_path="captured_image.jpg", data_path="data.pt")
print(person)
print(difference_list)