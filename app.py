from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from predict_score import get_prediction

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        score = get_prediction(img_path, user_tags)
        
        df = pd.read_csv("dress_dataset.csv")
        df['id'] = df['id'].apply(lambda x: str(x).zfill(4))

        # --- 分層推薦邏輯 ---
        rec_list = []
        
        # 1. 【高分挑戰】使用較寬鬆的過濾 (只看 性別+季節+年齡)
        challenge_pool = df[
            (df['gender'] == user_tags['gender']) & 
            (df['season'] == user_tags['season']) &
            (df['age'] == user_tags['age'])
        ]
        
        # 如果使用者分數低於 8.5，找一個比他高的
        if score < 8.5:
            higher_df = challenge_pool[challenge_pool['score'] > score].sort_values(by='score', ascending=False)
            if not higher_df.empty:
                top_one = higher_df.iloc[0]
                rec_list.append({
                    'id': top_one['id'], 'score': round(top_one['score'], 2),
                    'img_path': top_one['img_path'], 'label': '高分挑戰'
                })

        # 2. 【風格參考】使用最嚴格過濾 (五項全中)
        strict_pool = df[
            (df['gender'] == user_tags['gender']) & 
            (df['season'] == user_tags['season']) &
            (df['age'] == user_tags['age']) &
            (df['body'] == user_tags['body']) &
            (df['formal'] == user_tags['formal'])
        ]

        # 排除已選入挑戰位的 ID
        used_ids = [item['id'] for item in rec_list]
        
        # 填充剩餘的 2 個名額
        remaining_needed = 3 - len(rec_list)
        suit_df = strict_pool[~strict_pool['id'].isin(used_ids)].sort_values(by='score', ascending=False)
        
        # 如果嚴格過濾不夠 3 套，瀑布式向下填補
        if len(suit_df) < remaining_needed:
            backup_df = challenge_pool[~challenge_pool['id'].isin(used_ids)].sort_values(by='score', ascending=False)
            suit_df = pd.concat([suit_df, backup_df]).drop_duplicates(subset='id')

        for _, row in suit_df.head(remaining_needed).iterrows():
            rec_list.append({
                'id': row['id'], 'score': round(row['score'], 2),
                'img_path': row['img_path'], 'label': '風格參考'
            })

        return jsonify({
            'score': score, 'image_url': img_path, 'tags': user_tags, 'recommendations': rec_list
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9528, debug=True)