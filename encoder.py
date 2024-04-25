import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.models import vgg16
from torchvision import datasets
from torch.utils.data import DataLoader
from threading import Thread
from helper import timing
import argparse
import shutil
import os
from config import DATASET_PATH, VGG_DATA_PATH, STAGE_PATH, RESNET_DATA_PATH, VGG_WEIGHTS_PATH

# === ArgParse ===
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--isNew', action='store_true', help='Whether to process new data')
args = parser.parse_args()
isNew = args.isNew

# === Configuration ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === MTCNN and RESNET ===
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
vgg_model = vgg16(pretrained=False)
vgg_model.load_state_dict(torch.load(VGG_WEIGHTS_PATH))
vgg_model.eval().to(device)

# === Load old embeddings ===
if isNew:
    old_data_vgg = torch.load(VGG_DATA_PATH)
    old_embedding_list_vgg, old_name_list = old_data_vgg
    old_data_resnet = torch.load(RESNET_DATA_PATH)
    old_embedding_list_resnet, old_name_list = old_data_resnet
else:
    old_embedding_list_vgg = []
    old_embedding_list_resnet = []
    old_name_list = []


@timing
def get_embed_data(isNew, model):
    if model == 'vgg':
        dataset_path = STAGE_PATH if isNew else DATASET_PATH
        dataset = datasets.ImageFolder(dataset_path)
        idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        model = vgg_model
    elif model == 'resnet':
        dataset_path = STAGE_PATH if isNew else DATASET_PATH
        dataset = datasets.ImageFolder(dataset_path)
        idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        model = resnet

    def collate_func(batch):
        faces = []
        indices = []
        threads = []

        def process_image(img, idx):
            face, prob = mtcnn(img, return_prob=True)
            if face is not None and prob >= 0.9:
                faces.append(face)
                indices.append(idx)

        for img, idx in batch:
            thread = Thread(target=process_image, args=(img, idx))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        return faces, indices

    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_func)
    embeddings_dict = {}

    for batch_faces, batch_indices in loader:
        # Move faces to device before processing
        faces = [face.to(device) for face in batch_faces]
        with torch.no_grad():
            faces = [face.unsqueeze(0).to(device) for face in batch_faces]
            if model == vgg_model:
                features = torch.cat(faces)
                embeddings = vgg_model(features)
                embeddings = embeddings.view(embeddings.size(0), -1).cpu()
            elif model == resnet:
                embeddings = resnet(torch.cat(faces)).cpu()
                
        for idx, emb in zip(batch_indices, embeddings):
            if idx not in embeddings_dict:
                embeddings_dict[idx] = [emb]
            else:
                embeddings_dict[idx].append(emb)

    embedding_list = []
    name_list = []
    for idx, embeddings_list in embeddings_dict.items():
        avg_embedding = torch.stack(embeddings_list).mean(dim=0)
        embedding_list.append(avg_embedding)
        name_list.append(idx_to_class[idx])

    return embedding_list, name_list

embedding_list_resnet, name_list = get_embed_data(isNew, 'resnet')
embedding_list_vgg, name_list = get_embed_data(isNew, 'vgg')

if isNew:
    old_embedding_list_vgg.extend(embedding_list_vgg)
    old_name_list.extend(name_list)
    embedding_list_vgg, name_list = old_embedding_list_vgg, old_name_list
    old_embedding_list_resnet.extend(embedding_list_resnet)
    embedding_list_resnet, name_list = old_embedding_list_resnet, old_name_list

data_vgg = [embedding_list_vgg, name_list]
data_resnet = [embedding_list_resnet, name_list]

print(f"New name_list: {name_list}")

def save_data(model):
    if model == 'vgg':
        data = data_vgg
        print(f"Save {VGG_DATA_PATH}.")
        torch.save(data, VGG_DATA_PATH)
    elif model == 'resnet':
        data = data_resnet
        print(f"Save {RESNET_DATA_PATH}.")
        torch.save(data, RESNET_DATA_PATH)

@timing
def moving():
    f"""
    Move folders from {STAGE_PATH} to {DATASET_PATH} after embedding new students in {STAGE_PATH}
    """

    new_folders = os.listdir(STAGE_PATH)
    new_folders_path = [os.path.join(STAGE_PATH, folder) for folder in new_folders]
    destination_path = DATASET_PATH
    for source_path in new_folders_path:
        shutil.move(source_path, destination_path)
        print(f"Moved {source_path} => {destination_path}")

save_data('vgg')
save_data('resnet')
if isNew:
    moving()