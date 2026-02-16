import os
import pandas as pd
import re

RANK_DIR = "./static/for_ranking/"
CSV_PATH = "dress_dataset.csv"

def sync_scores_from_filenames():
    df = pd.read_csv(CSV_PATH, dtype={'id': str})
    
    # 掃描資料夾
    files = [f for f in os.listdir(RANK_DIR) if f.startswith('rank')]
    
    for filename in files:
        # 使用正則表達式精準抓取 S 後面的數字 和 ID 後面的數字
        # 範例檔名: rank0011a_S8.45_ID0999.jpg
        score_match = re.search(r'_S(\d+\.\d+)_', filename)
        id_match = re.search(r'_ID(\d+)', filename)
        
        if score_match and id_match:
            new_score = float(score_match.group(1))
            img_id = id_match.group(1).zfill(4)
            
            # 更新 CSV 中的分數
            df.loc[df['id'] == img_id, 'score'] = new_score

    df.to_csv(CSV_PATH, index=False)
    print("✨ 已將資料夾中的手動分數同步至 CSV。")

if __name__ == "__main__":
    sync_scores_from_filenames()