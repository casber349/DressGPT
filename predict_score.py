import torch
import torch.nn as nn
import clip
from PIL import Image
import os  # 新增：用於處理路徑

from feature_utils import get_one_hot_tags

# 1. 定義模型架構 (必須與 train_DressGPT.py 完全一致)
class DressGPT(nn.Module):
    def __init__(self):
        super(DressGPT, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(527, 256), 
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# 2. 預測函式 (改為 Ensemble 版本)
def get_prediction(image_path, user_tags):
    """
    使用 5-Fold Ensemble 模型進行平均預測
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = "./DressGPT_models"  # 設定模型資料夾路徑
    
    # A. 提取圖片與標籤特徵 (這部分只需要做一次，不用重複做 5 次)
    # ---------------------------------------------------------
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ 圖片讀取錯誤: {e}")
        return 0.0

    with torch.no_grad():
        # 取得 CLIP 圖片向量
        img_feat = clip_model.encode_image(image).to(torch.float32)
        img_feat /= img_feat.norm(dim=-1, keepdim=True) # 正規化
        
        # 取得標籤 One-hot 向量
        tag_feat = get_one_hot_tags(user_tags).to(device).unsqueeze(0)
        
        # 拼接成最終輸入特徵 (527維)
        combined_feat = torch.cat([img_feat, tag_feat], dim=1)

    # B. 載入 5 個模型並進行集成預測 (Ensemble Prediction)
    # ---------------------------------------------------------
    total_score = 0.0
    models_loaded = 0
    
    print(f"🔄 開始 5-Fold Ensemble 預測...")
    
    for i in range(1, 6):
        model_path = os.path.join(model_dir, f"fold{i}.pth")
        
        if os.path.exists(model_path):
            # 建立模型實例
            model = DressGPT().to(device)
            # 載入權重
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval() # 記得切換到評估模式
            
            with torch.no_grad():
                # 預測分數
                score = model(combined_feat).item()
                total_score += score
                models_loaded += 1
                print(f"   - Fold {i}: {score:.2f}") # 除錯用，想看細節可以打開
        else:
            print(f"⚠️ 警告: 找不到模型檔案 {model_path}，跳過。")

    # C. 計算平均分數
    # ---------------------------------------------------------
    if models_loaded == 0:
        print("❌ 錯誤: 沒有載入任何模型，無法評分！")
        return 0.0
    
    avg_score = total_score / models_loaded
    final_score = max(0, min(10, round(avg_score, 2))) # 限制範圍 0~10 並取小數點後兩位
    
    print(f"✅ 最終評分: {final_score} (基於 {models_loaded} 個模型的平均)")
    
    return final_score