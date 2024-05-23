from helper import timing
from main import ImageInfo
from config import BENCHMARK_INFO_PATH, BENCHMARK_PATH
import json
import os

def get_indices(class_code):
    image_info = ImageInfo(class_code)
    true_ids = []
    for student in data:
        if class_code in student["Class Code"]:
            true_ids.append(student["ID"])

    indices = []
    for true_id in true_ids:
        for i, _id in enumerate(student_ids):
            if true_id == _id:
                indices.append(i)
    return indices, image_info

def get_id_and_path(class_code):
    indices, image_info = get_indices(class_code)
    _ids = [student_ids[i] for i in indices]
    _paths = [image_paths[i] for i in indices]
    return _ids, _paths, image_info

@timing
def calculate_accuracy(class_code):
    student_ids, image_paths, image_info = get_id_and_path(class_code)
    ids = []
    for path in image_paths:
        _id, _ = image_info.face_match(path)
        ids.append(_id)
    accuracy = sum([1 for id1, id2 in zip(ids, student_ids) if id1 == id2]) / len(ids)
    return accuracy, len(ids)

def main(start_index, num_target_students, start_index_id):
    num_students = 40
    for i in range(len(data[start_index:start_index+num_target_students])):
        class_code = f"SE{start_index_id + i%num_students}"
        accuracy, num_images = calculate_accuracy(class_code)
        print(f"{class_code} --- {num_images} images --- Accuracy: {(accuracy * 100):.4f}%")
    
    return (start_index + num_students), (start_index_id + num_target_students)

if __name__ == "__main__":
    image_paths = []
    for root, dirs, files in os.walk(BENCHMARK_PATH):
        for file_name in files:
            image_path = os.path.join(root, file_name)
            image_paths.append(image_path)
    student_ids =[image_path.split("/")[1] for image_path in image_paths]

    with open(BENCHMARK_INFO_PATH, 'r') as f:
        data = json.load(f)
    
    start_index = 0
    start_index_id = 1900
    num_target_students = 1
    for _ in range(3):
        start_index, start_index_id = main(start_index, num_target_students, start_index_id)
