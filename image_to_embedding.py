import torch
import clip
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

# 1. 設定
CSV_PATH = "dress_dataset.csv"
OUTPUT_PT = "image_embeddings.pt" # 最終儲存向量的檔案
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在使用裝置: {device}")

# 2. 載入 CLIP 模型
model, preprocess = clip.load("ViT-B/32", device=device)

# 3. 讀取 CSV
df = pd.read_csv(CSV_PATH)
# 確保 ID 是字串且補零 (防呆)
df['id'] = df['id'].apply(lambda x: str(x).zfill(4))

embeddings_list = []
valid_ids = []

print("開始提取圖片特徵...")

# 4. 批次處理圖片
# 我們一張一張處理，避免記憶體溢位 (OOM)
with torch.no_grad():
    for index, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['id']
        img_path = row['img_path']
        
        if not os.path.exists(img_path):
            print(f"\n警告: 找不到圖片 {img_path}，跳過此筆。")
            continue
            
        try:
            # 讀取並預處理圖片
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            
            # 提取特徵向量 (Image Encoder)
            image_features = model.encode_image(image)
            
            # 正規化 (讓向量長度為 1，這對模型收斂很有幫助)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            embeddings_list.append(image_features.cpu())
            valid_ids.append(img_id)
            
        except Exception as e:
            print(f"\n處理 {img_id} 時發生錯誤: {e}")

# 5. 合併並儲存
if embeddings_list:
    # 將 List 轉為 [N, 512] 的大 Tensor
    all_embeddings = torch.cat(embeddings_list, dim=0)
    
    # 儲存為字典，包含向量與對應的 ID
    torch.save({
        'ids': valid_ids,
        'embeddings': all_embeddings
    }, OUTPUT_PT)
    
    print(f"\n成功！所有特徵已儲存至 {OUTPUT_PT}")
    print(f"最終特徵矩陣形狀: {all_embeddings.shape}")
else:
    print("\n沒有任何特徵被提取。")