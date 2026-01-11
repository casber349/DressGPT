import pandas as pd

def get_benchmarks(file_path='dress_dataset.csv'):
    # 強制讀取 id 為字串，避免被自動轉型
    df = pd.read_csv(file_path, dtype={'id': str})
    df['id'] = df['id'].str.zfill(4)
    
    # 定義指標
    targets = {
        "第一名": None,
        "+3 (9.50)": 9.50,
        "+2 (8.00)": 8.00,
        "+1 (6.50)": 6.50,
        "正中間 (5.00)": 5.00,
        "-1 (3.50)": 3.50,
        "-2 (2.00)": 2.00,
        "-3 (0.50)": 0.50,
        "最後一名": None
    }
    
    print("=== DressGPT 審美對標指標 ===")
    
    for label, target in targets.items():
        if label == "第一名":
            top = df.loc[df['score'].idxmax()]
            print(f" {label}: {top['id']} ({top['score']}分)")
            
        elif label == "最後一名":
            bottom = df.loc[df['score'].idxmin()]
            print(f" {label}: {bottom['id']} ({bottom['score']}分)")
            
        elif label == "+3 (9.50)":
            high_samples = df[df['score'] >= 9.50]
            if high_samples.empty:
                # 找不到 +3，列出所有 +2 (8.0) 以上的
                alt_samples = df[df['score'] >= 8.00].sort_values('score', ascending=False)
                ids_with_scores = [f"{row['id']}({row['score']})" for _, row in alt_samples.iterrows()]
                print(f" +2 以上: {', '.join(ids_with_scores)}")
            else:
                ids_with_scores = [f"{row['id']}({row['score']})" for _, row in high_samples.iterrows()]
                print(f" {label}: {', '.join(ids_with_scores)}")
                
        elif label == "-3 (0.50)":
            low_samples = df[df['score'] <= 0.50]
            if low_samples.empty:
                # 找不到 -3，列出所有 -2 (2.0) 以下的
                alt_samples = df[df['score'] <= 2.00].sort_values('score', ascending=True)
                ids_with_scores = [f"{row['id']}({row['score']})" for _, row in alt_samples.iterrows()]
                print(f" -2 以下: {', '.join(ids_with_scores)}")
            else:
                ids_with_scores = [f"{row['id']}({row['score']})" for _, row in low_samples.iterrows()]
                print(f" {label}: {', '.join(ids_with_scores)}")
        
        else:
            # 找出最接近目標值的 3 張圖
            df['diff'] = (df['score'] - target).abs()
            closest = df.sort_values('diff').head(3)
            # 依照分數排個序，看起來比較舒服
            closest = closest.sort_values('score', ascending=False)
            ids_with_scores = [f"{row['id']}({row['score']})" for _, row in closest.iterrows()]
            print(f" {label}: {', '.join(ids_with_scores)}")

if __name__ == "__main__":
    get_benchmarks()