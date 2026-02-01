import torch
import clip
import pandas as pd

TAG_MAPS = {
    "gender": {
        "options": [
            "a photo of a man, masculine facial features, short hair, male person", 
            "a photo of a woman, feminine facial features, long hair, wearing makeup, female person"
        ],
        "labels": ["male", "female"]
    },
    "age": {
        "options": [
            "a young teenager or child under 18 years old", 
            "a trendy young adult in their 20s or 30s", 
            "a mature middle-aged person in their 40s or 50s", 
            "a white-haired elderly person or senior citizen"
        ],
        "labels": ["teenager", "adult", "middle-aged", "elderly"]
    },
    "body": {
        "options": [
            "a very skinny thin body type with slender arms", 
            "a fit muscular athletic body with defined shape", 
            "a large plus size heavy body type, overweight", 
            "a normal average body type, neither thin nor fat"
        ],
        "labels": ["skinny", "athletic", "plus_size", "average"]
    },
    "season": {
        "options": [
            "wearing sleeveless tank top, shorts, or light t-shirt for hot summer", 
            "wearing very thick puffer jacket, heavy wool coat, scarf and gloves for cold winter", 
            "wearing a light jacket, hoodie, sweater or long sleeve shirt for spring or autumn"
        ],
        "labels": ["summer", "winter", "spring/fall"]
    },
    "formal": {
        "options": [
            "wearing a professional business suit, tuxedo, blazer and tie", 
            "wearing casual everyday clothes, street wear, t-shirt or hoodie"
        ],
        "labels": ["formal", "casual"]
    }
}

def run_auto_tagging(ids_list, embedding_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # 載入 Embedding 並確保是 float32
    all_embeddings = torch.load(embedding_path, map_location=device)
    
    # 預計算標籤向量
    tag_features = {}
    for attr, data in TAG_MAPS.items():
        with torch.no_grad():
            text_tokens = clip.tokenize(data["options"]).to(device)
            text_feats = clip_model.encode_text(text_tokens).to(torch.float32)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)
            tag_features[attr] = text_feats

    tagged_results = []
    print("🚀 正在重新精準執行視覺標註...")

    for img_id in ids_list:
        if img_id not in all_embeddings: continue
        
        # 取得圖片向量並轉為 float32
        img_feat = all_embeddings[img_id].to(device).to(torch.float32)
        if img_feat.ndim == 1: img_feat = img_feat.unsqueeze(0)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)

        res = {"id": img_id}
        for attr, text_feat in tag_features.items():
            # 這裡不使用 100.0 倍率，直接計算原始餘弦相似度
            similarity = (img_feat @ text_feat.T)
            top_idx = similarity.argmax().item()
            # 根據索引對應回我們定義的簡單標籤
            res[attr] = TAG_MAPS[attr]["labels"][top_idx]
            
        tagged_results.append(res)
        
    return pd.DataFrame(tagged_results)

import re
import torch.nn.functional as F
import image_to_embedding
import os

ENERGY_THRESHOLD_RATIO = 0.25

def infer_user_tags_via_neighbors(user_embed, csv_path, all_embeddings_path, k=10):
    """
    [God Mode] 完整輸出所有鄰居標籤與權重計算過程
    """
    df = pd.read_csv(csv_path)
    df['id'] = df['id'].astype(str)
    all_embeddings = torch.load(all_embeddings_path, map_location="cpu")
    
    ids = list(all_embeddings.keys())
    db_matrix = torch.stack([all_embeddings[i].flatten() for i in ids]).to(torch.float32)
    user_embed = user_embed.to(torch.float32).view(1, -1)
    
    # 相似度計算
    sims = F.cosine_similarity(user_embed, db_matrix)
    top_values, top_indices = torch.topk(sims.flatten(), k=min(k, len(ids)))
    
    tag_accumulator = {} 
    sim_sum_accumulator = {}

    print(f"\n" + "="*60)
    print(f"🕵️‍♂️ [深度審核] 正在分析相似度前 {k} 名的鄰居...")
    print(f"="*60)

    match_count = 0
    for i in range(len(top_indices)):
        idx = top_indices[i].item()
        sim = top_values[i].item()
        neighbor_id = str(ids[idx])
        
        # ID 處理邏輯
        try:
            clean_id = neighbor_id.split('.')[0]
            lookup_id = str(int(clean_id)) if clean_id.isdigit() else clean_id
        except:
            lookup_id = neighbor_id

        match_rows = df[df['id'].astype(str) == lookup_id]
        if match_rows.empty:
            match_rows = df[df['id'].astype(str) == neighbor_id]

        if not match_rows.empty:
            match_count += 1
            row = match_rows.iloc[0]
            # 取得該鄰居所有 pos_tags 與 neg_tags
            tags_str = f"{row.get('pos_tags', '')}, {row.get('neg_tags', '')}"
            matches = re.findall(r"\(([^:]+):([\d\.]+)\)", tags_str)
            
            print(f"📍 鄰居 {i+1}: ID={lookup_id.zfill(4)} | 相似度={sim:.4f}")
            print(f"   📜 原始標籤: {tags_str}") # 直接列印完整字串，不截斷

            for tag, weight in matches:
                w = float(weight)
                # 權重 * 相似度 累積
                tag_accumulator[tag] = tag_accumulator.get(tag, 0) + (w * sim)
                sim_sum_accumulator[tag] = sim_sum_accumulator.get(tag, 0) + sim
        else:
            print(f"⚠️ 鄰居 {neighbor_id} 在 CSV 中找不到，請檢查 ID 格式")

    print(f"\n" + "-"*60)
    print(f"🧮 梯度權重計算報告 (User Tag Inference):")
    print(f"-"*60)
    
    final_results = []
    for tag in tag_accumulator:
        total_energy = tag_accumulator[tag] # 這裡已經是 w * sim 的總和
        
        
        # 只有能量達標的才進入 final_results
        if total_energy >= ENERGY_THRESHOLD_RATIO * k:
            weighted_avg = round(total_energy / sim_sum_accumulator[tag], 1)
            clamped_w = max(0.1, min(1.9, weighted_avg))
            final_results.append((tag, clamped_w))
            print(f"保留{(tag, clamped_w)}標籤。標籤總強度:{total_energy}")
        else:
            print(f"捨棄{tag}標籤。標籤總強度:{total_energy}")
    
    final_results.sort(key=lambda x: x[1], reverse=True)

    # --- 在 auto_tagger.py 的 return final_results 前面加入 ---
    print("\n" + "!"*20 + " [核心算法診斷] " + "!"*20)
    print(f"總共參考了 {match_count} 個鄰居。")
    print("最終計算出的 User Tags (權重已完成相似度歸一化):")
    for tag, weight in final_results:
        # 這裡印出所有參與計算的標籤及其最終強度
        print(f" >> [標籤]: {tag:15s} | [最終權重]: {weight}")
    print("!"*55 + "\n")

    return final_results

# --- 終端機執行測試 ---
if __name__ == "__main__":
    # 請在此處輸入你想要測試的本地圖片路徑
    test_image_path = "my_gorgeous_friend.jpg" 
    db_csv = "dress_dataset.csv"
    db_emb = "image_embeddings.pt"

    if os.path.exists(test_image_path):
        print(f"🚀 開始對圖片 {test_image_path} 進行自動打標...")
        
        # 1. 提取 Embedding
        user_v = image_to_embedding.get_single_image_embedding(test_image_path)
        
        if user_v is not None:
            # 2. 進行推論
            inferred_tags = infer_user_tags_via_neighbors(user_v, db_csv, db_emb, k=8)
            
            print("\n" + "="*40)
            print(f"📊 DressGPT 自動打標報告 (19階強度)")
            print("="*40)
            for tag, weight in inferred_tags:
                # 權重高於 1.0 的標籤通常是顯性特徵
                star = "★" if weight > 1.2 else " "
                print(f"{star} {tag:18s} : {weight:.2f}")
            print("="*40)
        else:
            print("❌ 無法提取圖片特徵。")
    else:
        print(f"❌ 找不到測試圖片: {test_image_path}")