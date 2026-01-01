import os
import pandas as pd
from flask import Flask, render_template, request, redirect
import predict_score# 呼叫大腦

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# 在 app.py 中修改這行
DF = pd.read_csv('dress_dataset.csv', dtype={'id': str})

def get_tiered_recommendations(current_score, pool):
    """三階段推薦邏輯"""
    levels = [
        {"name": "微調方案", "range": (0.01, 1.00)},
        {"name": "進階方案", "range": (1.01, 2.50)},
        {"name": "極致改造", "range": (2.51, 10.00)}
    ]
    
    results = []
    for lv in levels:
        low, high = lv["range"]
        # 在符合過濾條件的池子中，尋找分數區間
        match = pool[(pool['score'] >= current_score + low) & (pool['score'] <= current_score + high)]
        
        if not match.empty:
            # 挑選該區間得分最高的
            best = match.sort_values(by='score', ascending=False).iloc[0].to_dict()
            best['level_label'] = lv["name"]
            results.append(best)
        else:
            results.append({"level_label": lv["name"], "id": "None", "score": "-", "prompt": "此區間暫無合適建議"})
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        u_gender = request.form.get('gender')
        u_season = request.form.get('season')
        u_formal = request.form.get('formal')

        if not file: return redirect(request.url)

        # 1. 統一存放到 static/uploads
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

        # 2. 統一呼叫大腦評分 (已處理 0-10 截斷)
        final_score = predict_score.predict_single_score(img_path)

        # 3. 根據標籤過濾資料庫 (Improver 邏輯)
        mask = (DF['gender'].isin([u_gender, 'both'])) & \
               (DF['season'] == u_season) & \
               (DF['formal_level'] == u_formal)
        filtered_pool = DF[mask]

        # 4. 取得三階段推薦
        recs = get_tiered_recommendations(final_score, filtered_pool)

        return render_template('index.html', 
                               user_img=file.filename,
                               score=final_score, 
                               recommendations=recs,
                               gender=u_gender, 
                               season=u_season, 
                               formal=u_formal)

    return render_template('index.html', score=None)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host="0.0.0.0", port=9528, debug=True)