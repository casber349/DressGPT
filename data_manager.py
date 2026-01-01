import pandas as pd
import os

# 設定路徑
PROMPTS_FILE = "./static/dataset_info/prompts.txt"
SCORES_FILE = "./static/dataset_info/scores.txt"
IMAGE_DIR = "./static/dataset_images/"
OUTPUT_CSV = "dress_dataset.csv"

def run_data_sync():
    print("正在整合原始數據與標籤...")
    
    # 1. 讀取原始資料
    prompts = {}
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                idx, content = line.strip().split(":", 1)
                prompts[idx.strip()] = content.strip()

    scores = {}
    with open(SCORES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                idx, val = line.strip().split(":", 1)
                scores[idx.strip()] = float(val.strip())

    # 2. 整合並自動判定標籤 (性別/季節/正式度)
    dataset = []
    for idx in sorted(prompts.keys()):
        p = prompts[idx].lower()
        
        # 性別
        gender = 'female' if any(w in p for w in ['woman', 'girl', 'lady']) else 'male'
        # 季節
        season = 'winter' if any(w in p for w in ['puffer', 'winter', 'coat', 'wool']) else \
                 'summer' if any(w in p for w in ['summer', 'shorts', 'linen']) else 'spring/fall'
        # 正式度
        formal = 'formal' if any(w in p for w in ['suit', 'blazer', 'office', 'shirt']) else 'casual'
        
        formatted_id = idx.zfill(4)
        dataset.append({
            "id": formatted_id,
            "score": scores.get(idx, 0.0),
            "prompt": prompts[idx],
            "gender": gender,
            "season": season,
            "formal_level": formal,
            "img_path": os.path.join(IMAGE_DIR, f"{formatted_id}.jpg")
        })

    df = pd.DataFrame(dataset)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 成功！已整合 {len(df)} 筆資料至 {OUTPUT_CSV}")

if __name__ == "__main__":
    run_data_sync()