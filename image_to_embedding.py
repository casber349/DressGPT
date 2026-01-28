import torch
import clip
from PIL import Image
import os
from tqdm import tqdm

# --- 設定 ---
IMAGE_DIR = "./static/dataset_images/"
OUTPUT_PT = "image_embeddings.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用裝置: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)

def get_single_image_embedding(img_path):
    """專門為單張上傳的照片提取 Embedding (512維)"""
    try:
        with torch.no_grad():
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            # 正規化向量，這樣計算 Cosine Similarity 時才準確
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu()
    except Exception as e:
        print(f"提取單張圖片特徵失敗: {e}")
        return None

def run_extraction():
    # 直接掃描資料夾內所有圖片
    img_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_files.sort()

    embedding_dict = {}

    print(f"開始提取 {len(img_files)} 張圖片的特徵...")
    with torch.no_grad():
        for filename in tqdm(img_files):
            # 取得 ID (例如 0001.jpg -> 0001)
            img_id = os.path.splitext(filename)[0]
            img_path = os.path.join(IMAGE_DIR, filename)
            
            try:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # 存入字典：{ "0001": tensor([...]), ... }
                embedding_dict[img_id] = image_features.cpu()
            except Exception as e:
                print(f"跳過錯誤圖片 {filename}: {e}")

    torch.save(embedding_dict, OUTPUT_PT)
    print(f"✅ 向量提取完成，已存至 {OUTPUT_PT}")

if __name__ == "__main__":
    run_extraction()