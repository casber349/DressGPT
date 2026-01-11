import pandas as pd
import numpy as np
from scipy.stats import norm

def normalize_dataset(file_path='dress_dataset.csv'):
    # 1. 讀取資料，強制將 id 視為字串
    df = pd.read_csv(file_path, dtype={'id': str})
    N = len(df)
    
    # 再次確保 ID 補齊四位 (預防之前被弄壞的資料)
    df['id'] = df['id'].str.zfill(4)
    
    # 2. 根據目前 score 欄位排序 (分數相同時依 ID 排序避免隨機性)
    df = df.sort_values(by=['score', 'id'], ascending=[False, True])
    
    # 3. 建立排名 (1 是最高分)
    df['rank'] = range(1, N + 1)
    
    # 4. 計算 PR 值 (0~100)
    # 排名第 1 的人 PR 會接近 100，最後一名接近 0
    df['pr'] = ((N - df['rank'] + 0.5) / N) * 100
    
    # 5. 將 PR 轉回 Z-score (需先轉回 0~1 比例)
    z_scores = norm.ppf(df['pr'] / 100)
    
    # 6. 計算標準化分數 (平均 5.0, 標準差 1.5) 並覆蓋原分數
    df['score'] = 5.0 + (z_scores * 1.5)
    df['score'] = df['score'].clip(0, 10).round(2)
    
    # 7. 移除不需要的欄位，保留 id, rank, pr, score
    if 'original_score' in df.columns:
        df = df.drop(columns=['original_score'])
    
    # 依照 ID 重新排回順序，方便閱讀
    df = df.sort_values('id')
    
    # 8. 存檔 (加上 quoting 確保字串不會被 Excel 亂轉，但最保險還是讀取時指定 dtype)
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    
    print(f"標準化完成！ID 已強制補正。")
    print(f"總樣本數: {N}")
    print(f"第一名: ID {df.loc[df['score'].idxmax(), 'id']} ({df['score'].max()} 分)")
    print(f"最後一名: ID {df.loc[df['score'].idxmin(), 'id']} ({df['score'].min()} 分)")

if __name__ == "__main__":
    normalize_dataset()