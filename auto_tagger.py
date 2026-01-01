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