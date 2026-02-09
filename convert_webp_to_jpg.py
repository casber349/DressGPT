from PIL import Image
import os

def batch_convert_webp_to_jpg(folder_path_src, folder_path_dst):
    for filename in os.listdir(folder_path_src):
        if filename.endswith(".webp"):
            img = Image.open(os.path.join(folder_path_src, filename)).convert("RGB")
            img.save(os.path.join(folder_path_dst, filename.replace(".webp", ".jpg")), "JPEG", quality=90)
            print(f"Converted {filename}")

batch_convert_webp_to_jpg('static/new_photos_for_dressgpt', 'static/dataset_images')