from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import torch
# 引入你原本的預測邏輯
from predict_score import get_prediction 
# 引入提取特徵的邏輯 (確保 image_to_embedding.py 有這個函式)
from image_to_embedding import get_single_image_embedding 
# 引入新寫的建議模組
from fashion_advisor import FashionAdvisor
from llm_consultant import DressConsultant

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 初始化建議系統
# 確保 image_embeddings.pt 與 dress_dataset.csv 路徑正確
advisor = FashionAdvisor(db_path='image_embeddings.pt', csv_path='dress_dataset.csv')
consultant = DressConsultant() # 金鑰會自動從 .env 讀取

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '沒有上傳檔案'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未選擇檔案'})

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    user_tags = {
        'gender': request.form.get('gender', 'male'),
        'age': request.form.get('age', 'adult'),
        'body': request.form.get('body', 'average'),
        'season': request.form.get('season', 'summer'),
        'formal': request.form.get('formal', 'casual')
    }

    try:
        # 1. 取得評分
        score = get_prediction(img_path, user_tags)
        
        # 2. 取得該張圖片的 Embedding
        # 使用你已經寫好的 image_to_embedding 邏輯
        user_embed = get_single_image_embedding(img_path) 

        # 3. 找出相似的高分與低分範例
        analysis_results = advisor.analyze(user_embed)
        
        # 4. 呼叫 Gemini 產生文字建議
        ai_advice = consultant.generate_advice(score, analysis_results)

        # --- 以下為你原本的推薦清單邏輯 (保持不變) ---
        df = pd.read_csv("dress_dataset.csv")
        df['id'] = df['id'].apply(lambda x: str(x).zfill(4))
        
        # (這裡省略你原本處理 rec_list 的過濾代碼...)
        # ... 原本的推薦邏輯 ...
        rec_list = [] # 假設這裡是產出的推薦清單

        # 5. 回傳所有結果給前端
        return jsonify({
            'score': score, 
            'image_url': img_path, 
            'tags': user_tags, 
            'recommendations': rec_list,
            'advice': ai_advice,  # LLM 的穿搭建議
            'analysis': {
                'good_ref': analysis_results['like_good_example'], # 像哪張好圖
                'bad_ref': analysis_results['like_bad_example']    # 像哪張壞圖
            }
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9528, debug=True)