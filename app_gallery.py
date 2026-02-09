import flask
from flask import request, jsonify, render_template
import pandas as pd
import os

app = flask.Flask(__name__)

# 設定 CSV 路徑
CSV_PATH = 'dress_dataset.csv'

@app.route("/")
def index():
    # 只負責回傳網頁骨架，不負責塞資料
    return render_template("gallery.html")

@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    try:
        # 1. 讀取 CSV (確保 id 是字串)
        if not os.path.exists(CSV_PATH):
            return jsonify({"error": "找不到 CSV 檔案"}), 404
            
        df = pd.read_csv(CSV_PATH, dtype={'id': str})
        df['id'] = df['id'].str.zfill(4) # 補零

        # 抓取全場天花板 (前四名)
        top_4_df = df.sort_values('score', ascending=False).head(4)
        top_four = []
        for _, r in top_4_df.iterrows():
            top_four.append({
                "id": r['id'], "score": round(r['score'], 2), 
                "img_path": r['img_path'], "tags": r.get('pos_tags', '')
            })

        # 抓取全場地板 (最後四名)
        bottom_4_df = df.sort_values('score', ascending=True).head(4)
        bottom_four = []
        for _, r in bottom_4_df.iterrows():
            bottom_four.append({
                "id": r['id'], "score": round(r['score'], 2), 
                "img_path": r['img_path'], "tags": r.get('neg_tags', '')
            })

        # 2. 定義我們要抓的分數錨點
        target_scores = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        results = []

        for target in target_scores:
            # 統一設定抓取 2 個最接近的樣本
            num_to_fetch = 2
            
            row_data = {
                "score": target,
                "male_list": [],
                "female_list": []
            }
            
            for gender in ['male', 'female']:
                sub_df = df[df['gender'] == gender].copy()
                if not sub_df.empty:
                    # 計算與目標分數的差距並取前 2 名
                    sub_df['diff'] = (sub_df['score'] - target).abs()
                    best_samples = sub_df.sort_values('diff').head(num_to_fetch)
                    
                    gender_key = f"{gender}_list"
                    for _, best in best_samples.iterrows():
                        row_data[gender_key].append({
                            "id": best['id'],
                            "score": round(best['score'], 2),
                            "img_path": best['img_path'],
                            "tags": f"{best.get('pos_tags', '')} {best.get('neg_tags', '')}"
                        })
            results.append(row_data)
        # 4. 回傳 JSON
        return jsonify({
            "benchmarks": results, 
            "top_four": top_four, 
            "bottom_four": bottom_four
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 使用 9529 port 避免跟原本的衝突
    app.run(host="127.0.0.1", port=9529, debug=True)