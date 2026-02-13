import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import os
import copy

from feature_utils import get_one_hot_tags
from model_arch import DressGPT

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
            combined_feat = torch.cat([img_feat, tag_feat]) 
            X_list.append(combined_feat)
            y_list.append(row['score'])

    if not X_list:
        raise ValueError("❌ 錯誤：沒有成功對齊任何資料！")

    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32).view(-1, 1)
    return X, y

def z_weighted_mse_loss(preds, targets):
    """
    分段式不對稱加權：低分區懲罰更狠
    """
    z_scores = (targets - 5.0) / 1.5
    abs_z = torch.abs(z_scores)
    
    # 創建基礎權重
    weights = 1.0
    
    # 針對低分區 (targets < 5.0) 額外加重
    # 使用 torch.where 進行分段判斷
    reward_multiplier = 0.6
    penalty_multiplier = 0.9  # 扣分狠度係數
    weights = torch.where(targets >= 5.0, 1.0 + reward_multiplier * abs_z, weights)
    weights = torch.where(targets < 5.0, 1.0 + penalty_multiplier * abs_z, weights)
    
    sq_errors = (preds - targets) ** 2
    return (weights * sq_errors).mean()

# 載入資料
X, y = load_and_prepare_data()

# 2. 5-Fold Ensemble 訓練
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_stats = [] 
epochs = 1000

print(f"🚀 開始 5-Fold Ensemble 訓練 (判定指標: Val_Loss)...")
print(f"{'Fold':<5} | {'Epoch':<6} | {'T-Loss':<8} | {'V-Loss':<8}")
print("-" * 65)

for fold, (t_idx, v_idx) in enumerate(kf.split(X)):
    X_t, X_v, y_t, y_v = X[t_idx], X[v_idx], y[t_idx], y[v_idx]
    
    model = DressGPT()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 初始化判定指標 (越小越好)
    best_v_loss = float('inf') 
    
    # 用來記錄最佳時刻的所有數據
    best_metrics = {}
    
    best_epoch = 0
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # 訓練流程
        train_output = model(X_t)
        loss = z_weighted_mse_loss(train_output, y_t)
        loss.backward()
        optimizer.step()
        
        # 驗證流程 (每 10 epoch 檢查一次)
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                y_t_pred = model(X_t)
                y_v_pred = model(X_v)
                
                # 1. 計算加權 Loss (核心指標)
                current_t_loss = z_weighted_mse_loss(y_t_pred, y_t).item()
                current_v_loss = z_weighted_mse_loss(y_v_pred, y_v).item()
                
                # 2. 計算 RMSE & R2 (參考指標)
                y_t_np, y_v_np = y_t.numpy(), y_v.numpy()
                pred_t_np, pred_v_np = y_t_pred.numpy(), y_v_pred.numpy()
                
                current_t_rmse = np.sqrt(mean_squared_error(y_t_np, pred_t_np))
                current_v_rmse = np.sqrt(mean_squared_error(y_v_np, pred_v_np))
                current_t_r2 = r2_score(y_t_np, pred_t_np)
                current_v_r2 = r2_score(y_v_np, pred_v_np)

                # 即時印出監控
                print(f"{fold+1:<5} | {epoch:<6} | {current_t_loss:.4f}   | {current_v_loss:.4f}")

                # 3. Early Stopping 判定 (只看 Val Loss)
                if current_v_loss < best_v_loss:
                    best_v_loss = current_v_loss
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    
                    # 暫存這一刻的所有數據以便最後寫入報告
                    best_metrics = {
                        'train_loss': current_t_loss,
                        'val_loss': current_v_loss,
                        'train_rmse': current_t_rmse,
                        'val_rmse': current_v_rmse,
                        'train_r2': current_t_r2,
                        'val_r2': current_v_r2
                    }
                else:
                    patience_counter += 1
            
            # 連續 15 次 (150 epochs) 沒有進步就停止
            if patience_counter >= 15:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
    
    # 儲存該 Fold 的最佳模型
    save_path = os.path.join(MODEL_SAVE_DIR, f"fold{fold+1}.pth")
    torch.save(best_model_wts, save_path)
    
    # 整理數據加入列表
    fold_stat = {
        "fold": fold + 1,
        "epoch": best_epoch,
        **best_metrics # 展開存入所有指標
    }
    fold_stats.append(fold_stat)
    print(f"✅ Fold {fold+1} Finished. Best V-Loss: {best_v_loss:.4f}\n")
    print("-" * 65)

# 3. 輸出報告 (格式：train_loss, val_loss, train_RMSE, val_RMSE, train_R2, val_R2)
print(f"📝 正在寫入訓練報告至 {INFO_FILE_PATH}...")

# 計算平均值
avg_t_loss = np.mean([s['train_loss'] for s in fold_stats])
avg_v_loss = np.mean([s['val_loss'] for s in fold_stats])
avg_t_rmse = np.mean([s['train_rmse'] for s in fold_stats])
avg_v_rmse = np.mean([s['val_rmse'] for s in fold_stats])
avg_t_r2 = np.mean([s['train_r2'] for s in fold_stats])
avg_v_r2 = np.mean([s['val_r2'] for s in fold_stats])

with open(INFO_FILE_PATH, "w", encoding="utf-8") as f:
    f.write("=== DressGPT v10 Z-Weighted Training Report ===\n\n")
    # 定義標題欄位
    header = f"{'Fold':<5} | {'Epoch':<6} | {'T-Loss':<8} | {'V-Loss':<8} | {'T-RMSE':<8} | {'V-RMSE':<8} | {'T-R2':<8} | {'V-R2':<8}\n"
    f.write(header)
    f.write("-" * len(header) + "\n")
    
    for s in fold_stats:
        line = (f"{s['fold']:<5} | {s['epoch']:<6} | "
                f"{s['train_loss']:.4f}   | {s['val_loss']:.4f}   | "
                f"{s['train_rmse']:.4f}   | {s['val_rmse']:.4f}   | "
                f"{s['train_r2']:.4f}   | {s['val_r2']:.4f}\n")
        f.write(line)
    
    f.write("-" * len(header) + "\n")
    avg_line = (f"{'AVG':<5} | {'-':<6} | "
                f"{avg_t_loss:.4f}   | {avg_v_loss:.4f}   | "
                f"{avg_t_rmse:.4f}   | {avg_v_rmse:.4f}   | "
                f"{avg_t_r2:.4f}   | {avg_v_r2:.4f}\n")
    f.write(avg_line)