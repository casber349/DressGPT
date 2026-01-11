import os
import time
import base64
import io
import torch
import pandas as pd
from flask import Flask, render_template, request, jsonify
from PIL import Image

# 引入自定義模組
from predict_score import get_prediction 
from image_to_embedding import get_single_image_embedding 
from fashion_advisor import FashionAdvisor
from llm_consultant import DressConsultant
from inpaint_engine import InpaintEngine

app = Flask(__name__)

# --- 1. 設定結構化目錄 ---
BASE_UPLOAD_PATH = 'static/uploads'
PATHS = {
    'orig': os.path.join(BASE_UPLOAD_PATH, 'originals'),
    'mask': os.path.join(BASE_UPLOAD_PATH, 'masks'),
    'result': os.path.join(BASE_UPLOAD_PATH, 'results')
}

# 自動建立所有必要目錄
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# --- 2. 初始化核心引擎 ---
# 注意：在啟動時載入，避免每次 request 都重新載入模型
advisor = FashionAdvisor(db_path='image_embeddings.pt', csv_path='dress_dataset.csv')
consultant = DressConsultant()
inpainter = InpaintEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '沒有上傳檔案'})
    
    file = request.files['file']
    mask_data = request.form.get('mask_image')
    timestamp = int(time.time())

    user_tags = {
        'gender': request.form.get('gender', 'male'),
        'age': request.form.get('age', 'adult'),
        'body': request.form.get('body', 'average'),
        'season': request.form.get('season', 'spring/fall'),
        'formal': request.form.get('formal', 'casual')
    }

    try:
        # 1. 處理原始圖片：儲存並調整尺寸
        img_filename = f"orig_{timestamp}.jpg"
        img_path = os.path.join(PATHS['orig'], img_filename)
        raw_img = Image.open(file.stream).convert("RGB")
        fixed_img = raw_img.resize((576, 1024), Image.LANCZOS)
        fixed_img.save(img_path)

        # 2. [重要] 無論有無重繪，先對原圖做基礎分析
        # 這樣才能拿到 analysis_results 用來生成動態 Prompt
        user_embed = get_single_image_embedding(img_path)
        # 傳入 user_tags 供 analyze 進行性別過濾
        analysis_results = advisor.analyze(user_embed, user_tags)

        final_image_path = img_path
        is_inpainted = False

        # 3. 判斷遮罩是否有內容 (防錯保護)
        if mask_data and "," in mask_data:
            header, encoded = mask_data.split(",", 1)
            mask_bytes = base64.b64decode(encoded)
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L").resize((576, 1024))
            
            # 使用 getbbox() 檢查遮罩是否有白色區域 (非全黑)
            if mask_img.getbbox():
                mask_filename = f"mask_{timestamp}.png"
                mask_path = os.path.join(PATHS['mask'], mask_filename)
                mask_img.save(mask_path)
                
                # get_inpaint_configs 會自動根據 user_tags 生成三段式 Prompt
                target_prompt, neg_prompt = advisor.get_inpaint_configs(analysis_results, user_tags)
                
                print(f"🎨 [AI 重繪處方箋]\n🔥 Positive: {target_prompt}\n🚫 Negative: {neg_prompt}")
                
                # 執行重繪
                inpainted_img = inpainter.generate(img_path, mask_path, target_prompt, neg_prompt)
                
                res_path = os.path.join(PATHS['result'], f"res_{timestamp}.jpg")
                inpainted_img.save(res_path)
                
                final_image_path = res_path
                is_inpainted = True
                
                # 重繪後重新分析新圖，獲取最終分數
                user_embed = get_single_image_embedding(final_image_path)
                # 傳入 user_tags 供 analyze 進行性別過濾
                analysis_results = advisor.analyze(user_embed, user_tags)
            else:
                print("⚠️ 警告：偵測到空遮罩，跳過重繪直接分析原圖。")

        # 4. 進行最終評分
        score = get_prediction(final_image_path, user_tags)

        # 在「5. 產生 AI 穿搭建議」之前加入以下邏輯：
        # 從 advisor 的資料庫中提取鄰居的原始標籤字串
        good_row = advisor.df[advisor.df['id_str'] == analysis_results['good_id']].iloc[0]
        bad_row = advisor.df[advisor.df['id_str'] == analysis_results['bad_id']].iloc[0]

        # 將標籤存入，供 consultant 使用
        analysis_results['good_tags'] = good_row.get('pos_tags', "無標籤數據")
        analysis_results['bad_tags'] = bad_row.get('neg_tags', "無標籤數據")
        
        # 5. 產生 AI 穿搭建議
        try:
            ai_advice = consultant.generate_advice(score, analysis_results, is_inpainted=is_inpainted)
        except Exception as e:
            print(f"⚠️ Gemini API 呼叫失敗: {e}")
            # API 失敗時，自動切換至顯示原始標籤數據的備用方案
            ai_advice = consultant.generate_backup_advice(score, analysis_results)

        return jsonify({
            'score': round(float(score), 2), 
            'image_url': final_image_path,
            'advice': ai_advice,
            'analysis': {
                'good_ref': analysis_results['like_good_example'],
                'bad_ref': analysis_results['like_bad_example']
            }
        })

    except Exception as e:
        print(f"❌ 系統錯誤: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # 禁止 reloader 以免載入兩次 SD 模型炸顯存
    app.run(host="0.0.0.0", port=9528, debug=True, use_reloader=False)