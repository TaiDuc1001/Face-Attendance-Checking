import pickle

ANNOTATION_TRAIN_PATH = "annotations/wider_face_train_bbx_gt.txt" 
ANNOTATION_VAL_PATH = "annotations/wider_face_val_bbx_gt.txt" 

def process_annotation_file(annotation_file_path):
	annotations = []
	with open(annotation_file_path, "r") as f:
		lines = f.readlines()
		i = 0
		while i < len(lines):
			# Take out the image path
			image_path = lines[i].strip()
			i += 1
			
			# Take out number of faces
			num_faces = int(lines[i].strip())
			i += 1
			
			# In case there is no faces in image (num_faces = 0)
			if num_faces == 0:
				num_faces += 1
				
			# Process all faces
			face_annotations = []
			for j in range(num_faces):
				face_attributes = lines[i + j].strip().split()
				x, y, w, h = map(int, face_attributes[:4])
				blur, expression, illumination, invalid, occlusion, pose = map(int, face_attributes[4:])
				face_annotations.append({
					"x": x,
					"y": y,
					"w": w,
					"h": h,
					"blur": blur,
					"expression": expression,
					"illumination": illumination,
					"invalid": invalid,
					"occlusion": occlusion,
					"pose": pose
				})
				
			# Append all informations to a dictionary annotations
			annotations.append({
				"Image path": image_path,
				"Face annotations": face_annotations
			})
			i += num_faces
	return annotations

# Dump train annotations
train_annotations = process_annotation_file(ANNOTATION_TRAIN_PATH)
pickle_train_path = "pickles/train_annotations.pickle"
with open(pickle_train_path, "wb") as pickle_file:
    pickle.dump(train_annotations, pickle_file)

# Dump val annotations
val_annotations = process_annotation_file(ANNOTATION_VAL_PATH)
pickle_val_path = "pickles/val_annotations.pickle"
with open(pickle_val_path, "wb") as pickle_file:
    pickle.dump(val_annotations, pickle_file)

print("Train annotations have been saved to:", pickle_train_path)
print("Val annotations have been saved to:", pickle_val_path)
