from helper import timing
from main import ImageInfo
import os
image_paths = []
for root, dirs, files in os.walk("benchmark"):
    for file_name in files:
        image_path = os.path.join(root, file_name)
        image_paths.append(image_path)

student_ids =[image_path.split("/")[1] for image_path in image_paths]
class ID_and_Paths():
    def __init__(self, student_ids, image_paths):
        self.ids = student_ids
        self.paths = image_paths
        self.len_ids = len(student_ids)
        self.len_paths = len(image_paths)
    def __len__(self):
        return len(self.ids) if self.len_ids == self.len_paths else "Length of IDs and Paths are not compatible!"
    def __getitem__(self, index):
        return [self.ids[index], self.paths[index]]

image_info = ImageInfo()
@timing
def test():
    ids = []
    index = 10
    for path in image_paths[:index]:
        _id, _ = image_info.face_match(path)
        ids.append(_id)

    accuracy = sum([1 for id1, id2 in zip(ids, student_ids) if id1 == id2]) / len(ids)
    return accuracy

accuracy = test()
print(f"Accuracy: {accuracy * 100}%")
