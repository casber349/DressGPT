import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import os
import copy

# 引入你的模組
from feature_utils import get_one_hot_tags
from model_arch import DressGPT

# 1. 設定路徑
CSV_PATH = "dress_dataset.csv"
EMBEDDINGS_PATH = "image_embeddings.pt"
MODEL_SAVE_DIR = "./DressGPT_models"
INFO_FILE_PATH = "./DressGPT_models/model_info.txt"
AUDIT_FILE_PATH = "./ensemble_validation_audit.csv"

# 確保儲存資料夾存在
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

def load_and_prepare_data():
    df = pd.read_csv(CSV_PATH)
    df['id'] = df['id'].apply(lambda x: str(x).zfill(4))
    id_to_feat = torch.load(EMBEDDINGS_PATH)
    
    X_list = []
    y_list = []
    valid_indices = []
    
    print("🔄 正在對齊圖片特徵與文字標籤...")
    for idx, row in df.iterrows():
        img_id = row['id']
        if img_id in id_to_feat:
            img_feat = id_to_feat[img_id].to(torch.float32).flatten()
            tag_feat = get_one_hot_tags(row)
            combined_feat = torch.cat([img_feat, tag_feat]) 
            
            X_list.append(combined_feat)
            y_list.append(row['score'])
            valid_indices.append(idx)

    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32).view(-1, 1)
    valid_df = df.loc[valid_indices].reset_index(drop=True)
    return X, y, valid_df

# ==========================================
# 核心改動：分層分桶 (Stratified Bucketing)
# ==========================================
def get_anchored_split(df, n_splits=5):
    """
    將 0-10 分切割為 10 個區段，確保每個 Fold 的驗證集在各區段的佔比均等
    """
    fold_ids = np.full(len(df), -1)
    
    # 定義桶子 (0-1, 1-2, ..., 9-10)
    # 使用 np.floor 將分數分類 (例如 9.45 分屬於第 9 桶)
    # 10 分會被分到第 10 桶，我們將其併入第 9 桶
    df['bucket'] = df['score'].apply(lambda x: min(int(np.floor(x)), 9))
    
    print(f"⚓ 執行全區段分層：正在將 1000 筆資料均勻分配至 {n_splits} 個 Fold...")

    for bucket_val in range(10):
        # 抓出該分數段的所有索引
        bucket_indices = df[df['bucket'] == bucket_val].index.to_numpy()
        np.random.shuffle(bucket_indices)
        
        # 將該桶子的索引輪流分配給各個 Fold
        for i, idx in enumerate(bucket_indices):
            fold_ids[idx] = i % n_splits
            
    # 檢查是否有漏網之魚
    if -1 in fold_ids:
        remaining = np.where(fold_ids == -1)[0]
        for i, idx in enumerate(remaining):
            fold_ids[idx] = i % n_splits

    for i in range(n_splits):
        val_mask = (fold_ids == i)
        yield np.where(~val_mask)[0], np.where(val_mask)[0]

def z_weighted_mse_loss(preds, targets):
    z_scores = (targets - 5.0) / 1.5
    abs_z = torch.abs(z_scores)
    # V11 追求泛化係數
    reward_multiplier = 0.6
    penalty_multiplier = 0.9
    weights = torch.where(targets >= 5.0, 1.0 + reward_multiplier * abs_z, 1.0)
    weights = torch.where(targets < 5.0, 1.0 + penalty_multiplier * abs_z, weights)
    return (weights * (preds - targets) ** 2).mean()

def print_distribution_health_check(df):
    total = len(df)
    
    # 1. 核心平庸區 (4.00 ~ 6.00)
    mid_zone = df[(df['val_score'] >= 4.00) & (df['val_score'] <= 6.00)]
    mid_count = len(mid_zone)
    
    # 2. 高分天堂區 (>= 8.00)
    high_zone = df[df['val_score'] >= 8.00]
    high_count = len(high_zone)
    
    # 3. 低分地獄區 (<= 2.00)
    low_zone = df[df['val_score'] <= 2.00]
    low_count = len(low_zone)
    
    print("\n" + "="*40)
    print("🚀 DressGPT V11 分佈健康檢查")
    print("="*40)
    print(f"1. 平庸區 (4.0-6.0): {mid_count:>4} 人 (預計 ~500) | 佔比: {mid_count/total:.1%}")
    print(f"2. 高分區 (>= 8.0): {high_count:>4} 人 (預計 ~20)  | 佔比: {high_count/total:.1%}")
    print(f"3. 低分區 (<= 2.0): {low_count:>4} 人 (預計 ~20)  | 佔比: {low_count/total:.1%}")
    print("="*40)
    
    # 異常警報邏輯
    if high_count < 5:
        print("⚠️ 警報：高分區人數太少！模型可能過於保守，加分加的不夠重。")
    if low_count < 5:
        print("⚠️ 警報：低分區人數太少！模型可能過於慈悲，扣分扣的不夠重。")
    if mid_count > total * 0.7:
        print("⚠️ 警報：模型給分太往中間偏。")
    if mid_count < total * 0.3:
        print("⚠️ 警報：模型給分太往兩邊偏。")

# ==========================================
# 主訓練流程
# ==========================================

X, y, valid_df = load_and_prepare_data()
splitter = get_anchored_split(valid_df, n_splits=5)
fold_stats = [] 
all_oof_results = [] # 儲存所有盲測結果

print(f"🚀 開始整合式 V11 訓練 (含全盲測稽核)...")

for fold, (t_idx, v_idx) in enumerate(splitter):
    X_t, X_v, y_t, y_v = X[t_idx], X[v_idx], y[t_idx], y[v_idx]
    model = DressGPT()
    # 加入 Weight Decay 強制泛化
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_v_loss = float('inf') 
    best_metrics = {}
    best_model_wts = None

    for epoch in range(1, 1001):
        model.train()
        optimizer.zero_grad()
        loss = z_weighted_mse_loss(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                p_t, p_v = model(X_t), model(X_v)
                curr_v_loss = z_weighted_mse_loss(p_v, y_v).item()

                if curr_v_loss < best_v_loss:
                    best_v_loss = curr_v_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    p_t_np, p_v_np = p_t.numpy(), p_v.numpy()
                    y_t_np, y_v_np = y_t.numpy(), y_v.numpy()
                    
                    best_metrics = {
                        'train_loss': z_weighted_mse_loss(p_t, y_t).item(),
                        'val_loss': curr_v_loss,
                        'train_rmse': np.sqrt(mean_squared_error(y_t_np, p_t_np)),
                        'val_rmse': np.sqrt(mean_squared_error(y_v_np, p_v_np)),
                        'train_r2': r2_score(y_t_np, p_t_np),
                        'val_r2': r2_score(y_v_np, p_v_np),
                        'best_epoch': epoch
                    }
                    patience = 0
                else:
                    patience += 1
            if patience >= 5: break

    # 儲存與記錄
    torch.save(best_model_wts, os.path.join(MODEL_SAVE_DIR, f"fold{fold+1}.pth"))
    fold_stats.append({"fold": fold+1, **best_metrics})
    
    # 紀錄殘差 (Residuals)
    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        v_preds = model(X_v).flatten().numpy()
        v_reals = y_v.flatten().numpy()
        v_ids = valid_df.iloc[v_idx]['id'].values
        for i in range(len(v_ids)):
            diff = abs(round((float(v_reals[i]) - float(v_preds[i])), 2))
            all_oof_results.append({
                'id': v_ids[i], 'real': round(float(v_reals[i]), 2),
                'val_score': round(float(v_preds[i]), 2),
                'diff': diff, 'which_fold': fold + 1
            })
    print(f"✅ Fold {fold+1} 完成. T-loss: {best_metrics['train_loss']:.4f}, V-loss: {best_metrics['val_loss']:.4f}")

# 3. 輸出 model_info.txt (不刪減任何欄位)
print(f"📝 正在寫入訓練報告至 {INFO_FILE_PATH}...")

avg_t_loss = np.mean([s['train_loss'] for s in fold_stats])
avg_v_loss = np.mean([s['val_loss'] for s in fold_stats])
avg_t_rmse = np.mean([s['train_rmse'] for s in fold_stats])
avg_v_rmse = np.mean([s['val_rmse'] for s in fold_stats])
avg_t_r2 = np.mean([s['train_r2'] for s in fold_stats])
avg_v_r2 = np.mean([s['val_r2'] for s in fold_stats])

with open(INFO_FILE_PATH, "w", encoding="utf-8") as f:
    f.write("=== DressGPT v11 Anchored Split Training Report ===\n\n")
    header = f"{'Fold':<5} | {'Epoch':<6} | {'T-Loss':<8} | {'V-Loss':<8} | {'T-RMSE':<8} | {'V-RMSE':<8} | {'T-R2':<8} | {'V-R2':<8}\n"
    f.write(header)
    f.write("-" * len(header) + "\n")
    
    for s in fold_stats:
        line = (f"{s['fold']:<5} | {s['best_epoch']:<6} | "
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

# 4. 輸出 ensemble_validation_audit.csv
audit_df = pd.DataFrame(all_oof_results).sort_values(by='id')

# 呼叫統計函式
print_distribution_health_check(audit_df)

audit_df.to_csv(AUDIT_FILE_PATH, index=False)
print(f"\n✨ 整合成功！\n1. 訓練報告已更新：{INFO_FILE_PATH}\n2. 全盲測稽核表已產出：{AUDIT_FILE_PATH}")