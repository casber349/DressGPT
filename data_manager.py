import pandas as pd
import os
import auto_tagger

PROMPTS_FILE = "./static/dataset_info/prompts.txt"
EMBEDDING_PATH = "image_embeddings.pt"
OUTPUT_CSV = "dress_dataset.csv"

def build_final_dataset():
    # 1. 檢查地基
    if not os.path.exists(EMBEDDING_PATH):
        print("❌ 錯誤：請先執行 image_to_embedding.py")
        return

    # 2. 讀取目前的 prompts.txt 作為最新名單
    prompts = {}
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                idx, content = line.strip().split(":", 1)
                prompts[idx.strip().zfill(4)] = content.strip()
    
    all_current_ids = sorted(list(prompts.keys()))

    # 3. 讀取現有的 CSV (如果有的話)
    existing_df = pd.DataFrame()
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        print(f"📂 偵測到現有資料集，包含 {len(existing_df)} 筆標註。")

    # 4. 找出需要新標註的 ID (在 prompts 裡但不在現有 CSV 裡)
    if not existing_df.empty:
        new_ids = [i for i in all_current_ids if i not in existing_df['id'].astype(str).str.zfill(4).values]
    else:
        new_ids = all_current_ids

    if not new_ids:
        print("✨ 沒有偵測到新圖片，資料集已是最新。")
        return

    print(f"🚀 發現 {len(new_ids)} 筆新資料 (從 {new_ids[0]} 到 {new_ids[-1]})，開始自動標註...")

    # 5. 只對新 ID 執行自動化標註
    new_tag_df = auto_tagger.run_auto_tagging(new_ids, EMBEDDING_PATH)
    
    # 6. 建立新資料的結構 (預留專家欄位)
    new_entries = []
    for _, row in new_tag_df.iterrows():
        img_id = row['id']
        new_entries.append({
            **row.to_dict(),
            "score": 0.0,       # 新資料預設 0 分
            "pos_tags": "",    # 待標註
            "neg_tags": "",    # 待標註
            "prompt": prompts.get(img_id, ""),
            "img_path": f"./static/dataset_images/{img_id}.jpg"
        })
    
    new_data_df = pd.DataFrame(new_entries)

    # 7. 合併前先確保 ID 格式統一
    if not existing_df.empty:
        # 強制將舊資料的 id 轉為四位字串
        existing_df['id'] = existing_df['id'].astype(str).str.zfill(4)
        
    # 強制將新資料的 id 轉為四位字串
    new_data_df['id'] = new_data_df['id'].astype(str).str.zfill(4)

    # 合併舊與新
    final_df = pd.concat([existing_df, new_data_df], ignore_index=True)
    
    # 現在排序就不會報錯了
    final_df = final_df.sort_values(by="id")

    # 8. 存檔 (加上 index=False)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 更新完成！目前資料總數：{len(final_df)}。")

if __name__ == "__main__":
    build_final_dataset()