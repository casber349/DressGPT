import os
import time
import base64
import io
import torch
import pandas as pd
from flask import Flask, render_template, request, jsonify
from PIL import Image
import math

# 引入自定義模組
from predict_score import get_prediction 
from image_to_embedding import get_single_image_embedding 
from fashion_advisor import FashionAdvisor
from llm_consultant import DressConsultant
from inpaint_engine import InpaintEngine

# --- 第 6 階段新增：引入診斷引擎 ---
from auto_tagger import infer_user_tags_via_neighbors

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
    # --- 改成這樣 ---
    mask_data = request.form.get('mask_image')
    last_result_path = request.form.get('last_result_path') 
    timestamp = int(time.time())
    
    # 定義變數，稍後填充
    img_path = ""

    user_tags = {
        'gender': request.form.get('gender', 'male'),
        'age': request.form.get('age', 'adult'),
        'body': request.form.get('body', 'average'),
        'season': request.form.get('season', 'spring/fall'),
        'formal': request.form.get('formal', 'casual')
    }

    # 獲取前端可能傳回來的「前一次結果路徑」
    last_result_path = request.form.get('last_result_path') 

    try:
        # 1. 路徑判定邏輯 (決定是用舊圖還是新圖)
        if last_result_path and os.path.exists(last_result_path):
            # [連續重繪模式]
            img_path = last_result_path
            print(f"🔄 [連續重繪模式] 使用前次結果: {img_path}")
        else:
            # [全新上傳模式] 這裡才檢查檔案是否存在
            if 'file' not in request.files:
                return jsonify({'error': '沒有上傳檔案，且無前次結果'})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': '檔案名稱為空'})

            img_filename = f"orig_{timestamp}.jpg"
            img_path = os.path.join(PATHS['orig'], img_filename)
            raw_img = Image.open(file.stream).convert("RGB")
            fixed_img = raw_img.resize((576, 1024), Image.LANCZOS)
            fixed_img.save(img_path)
            print(f"🆕 [全新上傳模式] 儲存原始圖片: {img_path}")

        # 2. [診斷階段] 提取 Embedding 與標籤診斷
        user_embed = get_single_image_embedding(img_path)
        user_diagnosis = infer_user_tags_via_neighbors(user_embed, 'dress_dataset.csv', 'image_embeddings.pt')
        
        # 預先取得原圖分數作為基準
        original_score = get_prediction(img_path, user_tags)

        # 3. [分析階段] 傳入診斷結果，獲取結構化分析報告
        # 注意：現在 analyze 必須傳入 user_diagnosis 才能生成 user_report
        analysis_results = advisor.analyze(user_embed, user_tags, user_diagnosis, original_score)

        if analysis_results is None:
            # 這是針對你提到的「找不到人就報錯」的處理
            return jsonify({
                'error': '數據庫中找不到符合您條件的對比範本，請嘗試更換照片或調整標籤。'
            }), 404

        final_image_path = img_path
        is_inpainted = False
        final_score = original_score

        # 4. 雙軌流程判定
        if mask_data and "," in mask_data:
            header, encoded = mask_data.split(",", 1)
            mask_bytes = base64.b64decode(encoded)
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L").resize((576, 1024))
            
            if mask_img.getbbox():
                # --- 有 Mask 流程 ---
                mask_filename = f"mask_{timestamp}.png"
                mask_path = os.path.join(PATHS['mask'], mask_filename)
                mask_img.save(mask_path)
                
                # A. 開藥：生成藥方
                target_p, neg_p = advisor.get_inpaint_configs(analysis_results, user_tags, user_diagnosis)

                # B. 計算動態強度 (方案 3)
                def get_real_pr(score):
                    """
                    將分數轉換為真實 PR 值 (基於常態分佈 Mu=5.0, Sigma=1.5)
                    """
                    mu = 5.0
                    sigma = 1.5
                    # 標準正態分佈的累積分布函數公式
                    pr = 0.5 * (1 + math.erf((score - mu) / (sigma * math.sqrt(2))))
                    return pr * 100

                def calculate_dynamic_strength_pr_real(score):
                    """
                    使用你設計的 PR 邏輯進行非線性強度轉換
                    """
                    pr = get_real_pr(score)
                    strength_100 = 0

                    # --- 你的專屬分段公式 ---
                    if score <= 3.5:
                        strength_100 = 100
                    elif score <= 6.0:
                        # 3.5~6.0 區間: 100 - 0.5 * PR
                        strength_100 = 100 - (0.5 * pr)
                    else:
                        # 6.0以上 區間: 2.5 * (100 - PR)
                        strength_100 = 2.5 * (100 - pr)

                    # 限制最小值，避免完全沒變化
                    final_strength = max(strength_100, 25)
                    
                    print(f"📊 [強度診斷] 分數: {score:.2f} | 真實 PR: {pr:.1f} | 最終強度: {final_strength/100:.2f}")
                    
                    return round(final_strength / 100, 2)
                
                # 替換成
                inpaint_strength = calculate_dynamic_strength_pr_real(original_score)
                print(f"🌡️ [手術室] 原圖分數: {original_score:.2f} | 預計強度: {inpaint_strength}")
                
                # C. 執行重繪：傳入動態強度
                # 注意：確保 generate 函式的參數順序與你 engine 定義一致
                inpainted_img = inpainter.generate(img_path, mask_path, target_p, neg_p, inpaint_strength)
                
                res_path = os.path.join(PATHS['result'], f"res_{timestamp}.jpg")
                inpainted_img.save(res_path)
                
                final_image_path = res_path
                is_inpainted = True
                
                # D. 術後驗收：對重繪後的圖進行最終評分
                final_score = get_prediction(final_image_path, user_tags)
                
                # --- 關鍵修正：重繪後需重新執行 analyze 以更新數據給 LLM ---
                # 這樣 LLM 才能知道「術後」的 user_report 有什麼變化
                new_user_embed = get_single_image_embedding(final_image_path)
                #analysis_results = advisor.analyze(new_user_embed, user_tags, user_diagnosis)
                analysis_results = advisor.analyze(new_user_embed, user_tags, user_diagnosis, final_score)
            else:
                print("⚠️ 偵測到空遮罩，進入「無 Mask 流程」...")

        # 5. 整合結果與 LLM 顧問諮詢
        # 確保數據結構完整，供新版 llm_consultant.py 使用
        analysis_results['original_score'] = original_score 

        ai_advice = consultant.generate_advice(final_score, analysis_results, is_inpainted=is_inpainted)

        return jsonify({
            'score': round(float(final_score), 2), 
            'original_score': round(float(original_score), 2),
            'image_url': final_image_path,
            'advice': ai_advice,
            'diagnosis': user_diagnosis,
            'analysis': {
                'good_ref': analysis_results['like_good_example'],
                'bad_ref': analysis_results['like_bad_example']
            }
        })

    except Exception as e:
        print(f"❌ 系統嚴重錯誤: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # 禁止 reloader 以免載入兩次 SD 模型炸顯存
    app.run(host="0.0.0.0", port=9528, debug=True, use_reloader=False)