import os
from PIL import Image

IMAGE_SIZE = 216
def resize_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
        img.save(image_path)
        print(f"Resized: {image_path}")
    except Exception as e:
        print(f"Error resizing {image_path}: {e}")

folder_path = "Images"

if os.path.exists(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(image_path)
                width, height = img.size
                if width != IMAGE_SIZE or height != IMAGE_SIZE:
                    resize_image(image_path)
                else:
                    print(f"Ignored (Already 216x216): {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
else:
    print("Images folder not found.")
