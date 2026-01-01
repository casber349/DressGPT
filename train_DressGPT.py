import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# 1. 設定路徑
CSV_PATH = "dress_dataset.csv"
EMBEDDINGS_PATH = "image_embeddings.pt"

# 2. 建立標籤轉換對照表 (將 CSV 的文字轉為數字，讓 AI 能計算)
GENDER_MAP = {"male": 0, "female": 1}
AGE_MAP = {"teenager": 0, "adult": 1, "middle-aged": 2, "elderly": 3}
BODY_MAP = {"skinny": 0, "athletic": 1, "plus_size": 2, "average": 3}
SEASON_MAP = {"summer": 0, "winter": 1, "spring/fall": 2}
FORMAL_MAP = {"formal": 0, "casual": 1}

def load_and_prepare_data():
    # 讀取 CSV 並確保 ID 格式正確 (如 0001)
    df = pd.read_csv(CSV_PATH)
    df['id'] = df['id'].apply(lambda x: str(x).zfill(4))
    
    # 💡 關鍵修正點：直接讀取字典格式 {id: tensor}
    id_to_feat = torch.load(EMBEDDINGS_PATH)
    
    X_list = []
    y_list = []
    valid_ids = []

    print("🔄 正在對齊圖片特徵與文字標籤...")
    for _, row in df.iterrows():
        img_id = row['id']
        # 現在直接從字典裡用 ID 領取向量
        if img_id in id_to_feat:
            # A. 取得 512 維圖片向量
            img_feat = id_to_feat[img_id].to(torch.float32).flatten()
            
            # B. 取得 5 維自定義標籤特徵 (使用對照表轉為數字)
            tag_feat = torch.tensor([
                GENDER_MAP.get(row.get('gender', 'male'), 0),
                AGE_MAP.get(row.get('age', 'adult'), 1),
                BODY_MAP.get(row.get('body', 'average'), 3),
                SEASON_MAP.get(row.get('season', 'summer'), 2),
                FORMAL_MAP.get(row.get('formal', 'casual'), 1)
            ], dtype=torch.float32)
            
            # C. 拼接特徵：512 (圖片) + 5 (標籤) = 517 維
            combined_feat = torch.cat([img_feat, tag_feat])
            
            X_list.append(combined_feat)
            y_list.append(row['score'])
            valid_ids.append(img_id)

    if not X_list:
        raise ValueError("❌ 錯誤：沒有成功對齊任何資料，請檢查 CSV 的 ID 與向量檔案是否匹配！")

    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32).view(-1, 1)
    return X, y, valid_ids

# 載入資料
X, y, ids = load_and_prepare_data()
print(f"✅ 載入成功！訓練樣本數: {len(X)}, 輸入總維度: {X.shape[1]}")

# 3. 定義模型 (輸入維度改為 517)
class DressGPT(nn.Module):
    def __init__(self):
        super(DressGPT, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(517, 256), 
            nn.ReLU(),
            nn.Dropout(0.2), # 增加穩定性
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

model = DressGPT()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 開始訓練
epochs = 1000
print(f"🚀 開始訓練 DressGPT (Deep Feature Fusion)...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X)
    loss = criterion(outputs, y)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 5. 儲存模型
torch.save(model.state_dict(), "dressgpt_weights.pth")
print("\n✅ 訓練完成！模型權重已儲存為 dressgpt_weights.pth")

# 6. 驗證前 5 筆預測
model.eval()
with torch.no_grad():
    preds = model(X[:5])
    print("\n--- 預測結果對比 ---")
    for i in range(min(5, len(ids))):
        print(f"ID: {ids[i]} | 實際分數: {y[i].item():.2f} | AI 預測: {preds[i].item():.2f}")