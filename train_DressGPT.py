import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import os
import copy  # 用來複製最佳模型參數

from feature_utils import get_one_hot_tags
from model_arch import DressGPT  # 從新檔案引入

# 1. 設定路徑
CSV_PATH = "dress_dataset.csv"
EMBEDDINGS_PATH = "image_embeddings.pt"
MODEL_SAVE_DIR = "./DressGPT_models"
INFO_FILE_PATH = "./DressGPT_models/model_info.txt"

# 確保儲存資料夾存在
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

def load_and_prepare_data():
    df = pd.read_csv(CSV_PATH)
    df['id'] = df['id'].apply(lambda x: str(x).zfill(4))
    id_to_feat = torch.load(EMBEDDINGS_PATH)
    
    X_list = []
    y_list = []
    
    print("🔄 正在對齊圖片特徵與文字標籤...")
    for _, row in df.iterrows():
        img_id = row['id']
        if img_id in id_to_feat:
            img_feat = id_to_feat[img_id].to(torch.float32).flatten()
            tag_feat = get_one_hot_tags(row)
            combined_feat = torch.cat([img_feat, tag_feat]) # 527 維
            X_list.append(combined_feat)
            y_list.append(row['score'])

    if not X_list:
        raise ValueError("❌ 錯誤：沒有成功對齊任何資料！")

    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32).view(-1, 1)
    return X, y

# 載入資料
X, y = load_and_prepare_data()

# 2. 5-Fold Ensemble 訓練
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_stats = [] # 用來存每個 fold 的數據
epochs = 1000

print(f"🚀 開始 5-Fold Ensemble 訓練 (將儲存 5 個模型)...")

for fold, (t_idx, v_idx) in enumerate(kf.split(X)):
    X_t, X_v, y_t, y_v = X[t_idx], X[v_idx], y[t_idx], y[v_idx]
    
    model = DressGPT()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_v_r2 = -float('inf')
    best_t_r2 = 0
    best_epoch = 0
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict()) # 初始化最佳權重

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        
        # 每 10 epoch 檢查一次
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                current_t_r2 = r2_score(y_t.numpy(), model(X_t).numpy())
                current_v_r2 = r2_score(y_v.numpy(), model(X_v).numpy())
                
                if current_v_r2 > best_v_r2:
                    best_v_r2 = current_v_r2
                    best_t_r2 = current_t_r2
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict()) # 📸 抓拍最佳權重
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience_counter >= 5: # Early Stop
                break
    
    # 儲存該 Fold 的最佳模型
    save_path = os.path.join(MODEL_SAVE_DIR, f"fold{fold+1}.pth")
    torch.save(best_model_wts, save_path)
    
    # 記錄數據
    fold_stat = {
        "fold": fold + 1,
        "epoch": best_epoch,
        "train_r2": best_t_r2,
        "val_r2": best_v_r2
    }
    fold_stats.append(fold_stat)
    
    print(f"✅ Fold {fold+1} 完成: Saved at Epoch {best_epoch} | Val R2: {best_v_r2:.4f}")

# 3. 輸出 model_info.txt
print(f"\n📝 正在寫入訓練報告至 {INFO_FILE_PATH}...")

avg_epoch = np.mean([s['epoch'] for s in fold_stats])
avg_train_r2 = np.mean([s['train_r2'] for s in fold_stats])
avg_val_r2 = np.mean([s['val_r2'] for s in fold_stats])

with open(INFO_FILE_PATH, "w", encoding="utf-8") as f:
    f.write("=== DressGPT 5-Fold Ensemble Report ===\n\n")
    f.write(f"{'Fold':<5} | {'Epoch':<6} | {'Train R2':<10} | {'Val R2':<10}\n")
    f.write("-" * 45 + "\n")
    
    for s in fold_stats:
        f.write(f"{s['fold']:<5} | {s['epoch']:<6} | {s['train_r2']:.4f}     | {s['val_r2']:.4f}\n")
    
    f.write("-" * 45 + "\n")
    f.write(f"{'AVG':<5} | {int(avg_epoch):<6} | {avg_train_r2:.4f}     | {avg_val_r2:.4f}\n")

print(f"🎉 訓練完成！所有模型與報告已儲存於 {MODEL_SAVE_DIR}")