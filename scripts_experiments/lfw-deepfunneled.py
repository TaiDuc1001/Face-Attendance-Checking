import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

from scripts.helper import timing
from scripts.compare_faces import ImageInfo
from scripts.config import BENCHMARK_INFO_PATH, BENCHMARK_PATH
import json
import argparse
from tqdm import tqdm

def get_indices(class_code, json_path=BENCHMARK_INFO_PATH):
    image_info = ImageInfo(class_code, json_path)
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
    for path in (image_paths):
        _id, _ = image_info.face_match(path, isBenchmark=True)
        ids.append(_id)
    accuracy = sum([1 for id1, id2 in zip(ids, student_ids) if id1 == id2]) / len(ids)
    return accuracy, len(ids)

def main(start_index, num_target_students, start_index_id):
    num_students = 40
    for i in range(len(data[start_index:start_index+num_target_students])):
        class_code = f"SE{start_index_id + i}"
        accuracy, num_images = calculate_accuracy(class_code)
        accuracies.append(accuracy)
        print(f"{class_code} --- {num_images} images --- Accuracy: {(accuracy * 100):.4f}%")
    
    return (start_index + num_students), (start_index_id + num_target_students)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark script.")
    parser.add_argument('--num_classes', type=int, help='The number of classes to run the benchmark on.')
    args = parser.parse_args()

    image_paths = []
    for root, dirs, files in os.walk(BENCHMARK_PATH):
        for file_name in files:
            image_path = os.path.join(root, file_name)
            image_paths.append(image_path)
    student_ids =[image_path.split("/")[1] for image_path in image_paths]
    print("Done scanning.")

    with open(BENCHMARK_INFO_PATH, 'r') as f:
        data = json.load(f)
    
    start_index = 0
    start_index_id = 1900
    num_target_students = args.num_classes if args.num_classes else (len(os.listdir(BENCHMARK_PATH)) // 40 + 1)
    print(f"Running the benchmark on {num_target_students} classes.")
    accuracies = []
    start_index, start_index_id = main(start_index, num_target_students, start_index_id)
    print(f"Average accuracy: {(sum(accuracies) / len(accuracies) * 100):.4f}%")
