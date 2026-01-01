import pandas as pd
import os
import auto_tagger

PROMPTS_FILE = "./static/dataset_info/prompts.txt"
SCORES_FILE = "./static/dataset_info/scores.txt"
EMBEDDING_PATH = "image_embeddings.pt"
OUTPUT_CSV = "dress_dataset.csv"

def build_final_dataset():
    if not os.path.exists(EMBEDDING_PATH):
        print("❌ 錯誤：請先執行 image_to_embedding.py")
        return

    # 1. 讀取文字資料
    print("Step 1: 讀取文字資料...")
    prompts = {}
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                idx, content = line.strip().split(":", 1)
                prompts[idx.strip().zfill(4)] = content.strip()

    scores = {}
    with open(SCORES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                idx, val = line.strip().split(":", 1)
                scores[idx.strip().zfill(4)] = float(val.strip())

    # 2. 以 Prompts 裡的 ID 為準，進行自動標註
    all_ids = list(prompts.keys())
    tag_df = auto_tagger.run_auto_tagging(all_ids, EMBEDDING_PATH)

    # 3. 合併所有資訊
    print("Step 2: 整合最終 CSV...")
    final_data = []
    for _, row in tag_df.iterrows():
        img_id = row['id']
        final_data.append({
            **row.to_dict(), # 包含 ID 和 AI 標註的屬性
            "score": scores.get(img_id, 0.0),
            "prompt": prompts.get(img_id, ""),
            "img_path": f"./static/dataset_images/{img_id}.jpg"
        })

    pd.DataFrame(final_data).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 成功！已建立資料集：{OUTPUT_CSV}")

if __name__ == "__main__":
    build_final_dataset()