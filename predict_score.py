import torch
import torch.nn as nn
import clip
from PIL import Image

# 1. 必須與訓練時完全相同的對照表
GENDER_MAP = {"male": 0, "female": 1}
AGE_MAP = {"teenager": 0, "adult": 1, "middle-aged": 2, "elderly": 3}
BODY_MAP = {"skinny": 0, "athletic": 1, "plus_size": 2, "average": 3}
SEASON_MAP = {"summer": 0, "winter": 1, "spring/fall": 2}
FORMAL_MAP = {"formal": 0, "casual": 1}

# 2. 定義模型架構 (必須與 train_DressGPT.py 一致)
class DressGPT(nn.Module):
    def __init__(self):
        super(DressGPT, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(517, 256), # 512 + 5 = 517
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# 3. 預測函式
def get_prediction(image_path, user_tags):
    """
    image_path: 使用者上傳的照片路徑
    user_tags: 字典格式，例如 {'gender': 'male', 'age': 'adult', ...}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # A. 載入模型與權重
    model = DressGPT().to(device)
    model.load_state_dict(torch.load("dressgpt_weights.pth", map_location=device))
    model.eval()
    
    # B. 提取圖片 CLIP 特徵 (512維)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        img_feat = clip_model.encode_image(image).to(torch.float32)
        img_feat /= img_feat.norm(dim=-1, keepdim=True) # 正規化
        
        # C. 處理使用者選取的標籤特徵 (5維)
        tag_feat = torch.tensor([
            GENDER_MAP.get(user_tags.get('gender'), 0),
            AGE_MAP.get(user_tags.get('age'), 1),
            BODY_MAP.get(user_tags.get('body'), 3),
            SEASON_MAP.get(user_tags.get('season'), 2),
            FORMAL_MAP.get(user_tags.get('formal'), 1)
        ], dtype=torch.float32).to(device).unsqueeze(0)
        
        # D. 拼接特徵並預測
        combined_feat = torch.cat([img_feat, tag_feat], dim=1)
        score = model(combined_feat).item()
        
    # 回傳 0-10 分的結果 (限制範圍)
    return max(0, min(10, round(score, 2)))