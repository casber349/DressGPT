import pandas as pd
import os
import shutil

# --- 設定區 ---
SOURCE_DIR = "./static/dataset_images/"
TARGET_DIR = "./static/for_ranking/"
CSV_PATH = "dress_dataset.csv"

def run_visual_audit():
    # 1. 強制清理並重建目標資料夾
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR)

    # 2. 讀取 CSV，強行限制 ID 為字串，避免自動轉型成 float
    df = pd.read_csv(CSV_PATH, dtype={'id': str})
    
    # 再次防禦：處理可能已經被轉壞的 0990.0 這種字串
    df['id'] = df['id'].apply(lambda x: x.split('.')[0].zfill(4) if pd.notna(x) else "0000")

    # 3. 排序邏輯：分數降序 (大牌在前)，ID 升序 (穩定排序)
    # 我們要把 9.94 分排在前面，0.5 分排在後面
    df_sorted = df.sort_values(by=['score', 'id'], ascending=[False, True]).reset_index(drop=True)

    print(f"🕵️ 正在整理排行榜，總計 {len(df_sorted)} 筆資料...")

    # 4. 批次處理
    for index, row in df_sorted.iterrows():
        rank = index + 1
        score = row['score']
        img_id = row['id']

        # 搜尋檔案
        src_exts = ['.jpg', '.webp', '.png', '.jpeg']
        found = False
        for ext in src_exts:
            potential_file = os.path.join(SOURCE_DIR, f"{img_id}{ext}")
            if os.path.exists(potential_file):
                # 構建新檔名：rank0001_S9.94_ID1000_male_formal.jpg
                new_filename = f"rank{str(rank).zfill(4)}_S{score:.2f}_ID{img_id}{ext}"
                shutil.copy2(potential_file, os.path.join(TARGET_DIR, new_filename))
                found = True
                break
        
        if not found:
            print(f"⚠️ 找不到 ID {img_id} 的對應圖檔")

    print(f"✅ 排行榜生成完畢！地點：{TARGET_DIR}")
    print(f"💡 小提示：請使用『大圖示』檢視，並按『名稱』排序。")

if __name__ == "__main__":
    run_visual_audit()