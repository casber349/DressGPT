import torch
import torch.nn as nn
import clip
from PIL import Image

from feature_utils import get_one_hot_tags

# 2. 定義模型架構 (必須與 train_DressGPT.py 一致)
class DressGPT(nn.Module):
    def __init__(self):
        super(DressGPT, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(527, 256), # 512 + 15 = 527
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
        
        # 這裡 user_tags 是前端傳來的 dict，直接丟進去
        tag_feat = get_one_hot_tags(user_tags).to(device).unsqueeze(0)
        
        # D. 拼接特徵並預測
        combined_feat = torch.cat([img_feat, tag_feat], dim=1)
        score = model(combined_feat).item()
        
    # 回傳 0-10 分的結果 (限制範圍)
    return max(0, min(10, round(score, 2)))