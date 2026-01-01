import torch
import clip
from PIL import Image
import os

# 配置
MODEL_PATH = "dressgpt_weights.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 模型架構 (必須與訓練時一致)
class DressGPT(torch.nn.Module):
    def __init__(self):
        super(DressGPT, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# 2. 預先載入模型與 CLIP (避免每次網頁點擊都重載，提升速度)
print("正在載入 AI 評分模型...")
model_net = DressGPT().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model_net.eval()
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

def predict_single_score(img_path):
    """專供網頁使用：對單張上傳照片進行評分"""
    try:
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = clip_model.encode_image(image).to(torch.float32)
            features /= features.norm(dim=-1, keepdim=True)
            score = model_net(features).item()
        
        # 解決爆表問題：強制限制在 0.00 - 10.00
        return round(max(0.0, min(10.0, float(score))), 2)
    except Exception as e:
        print(f"評分失敗: {e}")
        return 5.0 # 錯誤時的回退分數