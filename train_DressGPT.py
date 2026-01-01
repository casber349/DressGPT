import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# 1. 載入數據
CSV_PATH = "dress_dataset.csv"
EMBEDDINGS_PATH = "image_embeddings.pt"

# 載入評分
df = pd.read_csv(CSV_PATH)
y = torch.tensor(df['score'].values, dtype=torch.float32).view(-1, 1)

# --- 修改這部分 ---
# 載入圖片特徵
data = torch.load(EMBEDDINGS_PATH)
# 將 X 強制轉換為 float32，解決 Half vs Float 的衝突
X = data['embeddings'].to(torch.float32) 
# ------------------

print(f"數據載入完成！輸入形狀: {X.shape}, 目標形狀: {y.shape}")

# 2. 定義一個簡單但強大的線性回歸模型
class DressGPT(nn.Module):
    def __init__(self):
        super(DressGPT, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # 防止過擬合
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

model = DressGPT()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 訓練模型
epochs = 1000
print("開始訓練 DressGPT...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X)
    loss = criterion(outputs, y)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 4. 儲存模型權重
torch.save(model.state_dict(), "dressgpt_weights.pth")
print("\n訓練完成！模型權重已儲存為 dressgpt_weights.pth")

# 5. 簡單測試：看看前 5 張圖的預測與實際差異
model.eval()
with torch.no_grad():
    predictions = model(X[:5])
    for i in range(5):
        print(f"圖片 {data['ids'][i]} - 實際分數: {y[i].item():.2f}, 預測分數: {predictions[i].item():.2f}")