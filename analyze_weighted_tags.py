import pandas as pd
import re
import json

def analyze_weighted_trends(file_path='dress_dataset.csv'):
    try:
        # 強制讀取為字串避免 ID 跑掉，並確保編碼正確
        df = pd.read_csv(file_path, dtype={'id': str}, encoding='utf-8')
    except FileNotFoundError:
        print("找不到檔案，請確認路徑。")
        return

    # --- 修正後的核心正規表達式 ---
    # \(      : 匹配左括號
    # ([^:]+) : 擷取第一組（標籤名稱）：匹配所有「不是冒號」的字元（包含空格）
    # :       : 匹配冒號
    # ([\d.]+) : 擷取第二組（權重）：匹配數字與小數點
    # \)      : 匹配右括號
    pattern = r'\(([^:]+):([\d.]+)\)'

    def extract_weighted_tags(row):
        # 處理可能存在的空值，確保合併時為字串
        pos = str(row['pos_tags']) if pd.notnull(row['pos_tags']) else ""
        neg = str(row['neg_tags']) if pd.notnull(row['neg_tags']) else ""
        combined = f"{pos},{neg}"
        
        matches = re.findall(pattern, combined)
        # 這裡的 tag.strip() 可以移除標籤頭尾不小心留下的空白
        return [(tag.strip(), float(weight)) for tag, weight in matches]

    tag_data = []
    for _, row in df.iterrows():
        tags = extract_weighted_tags(row)
        for tag, weight in tags:
            tag_data.append({
                'tag': tag,
                'weight': weight,
                'score': row['score']
            })
    
    if not tag_data:
        print("未偵測到符合 (tag:weight) 格式的數據。")
        return

    w_df = pd.DataFrame(tag_data)
    
    # 核心統計
    stats = w_df.groupby('tag').agg(
        avg_weight=('weight', 'mean'),
        avg_score=('score', 'mean'),
        count=('score', 'count')
    ).reset_index()

    # 計算「有效影響力」（偏微分斜率）
    stats['effective_impact'] = (stats['avg_score'] - 5.0) / stats['avg_weight']
    stats = stats.sort_values('effective_impact', ascending=False)

    # --- 關鍵修改：匯出 JSON 藥力表 ---
    # 將 DataFrame 轉換為字典格式 {tag: impact}
    potency_map = dict(zip(stats['tag'], stats['effective_impact']))
    
    with open('labels_potency.json', 'w', encoding='utf-8') as f:
        json.dump(potency_map, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 藥理資料庫已儲存至 labels_potency.json，共紀錄 {len(potency_map)} 種藥性標籤。")
    # --- 結束修改 ---

    print(f"\n --- 所有標籤 (共 {len(stats)} 種) ---")
    print(stats[['tag', 'avg_weight', 'avg_score', 'effective_impact']])
    
    # 額外存成 CSV 方便你查看完整清單
    # stats.to_csv('final_impact_report.csv', index=False)

if __name__ == "__main__":
    analyze_weighted_trends()