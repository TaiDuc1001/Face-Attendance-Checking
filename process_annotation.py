ANNOTATION_TRAIN_PATH = "annotations\wider_face_train_bbx_gt.txt"

def read_annotation_file(annotation_file_path):
    annotations = []
    with open(annotation_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:  # Skip the first two lines (image dimensions)
            parts = line.strip().split()
            x, y, w, h = map(int, parts[:4])
            blur, expression, illumination, invalid, occlusion, pose = parts[4:]
            annotations.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'blur': blur,
                'expression': expression,
                'illumination': illumination,
                'invalid': invalid,
                'occlusion': occlusion,
                'pose': pose
            })
    return annotations

# Example usage
annotations = read_annotation_file(ANNOTATION_TRAIN_PATH)
for annotation in annotations:
    print(annotation)
