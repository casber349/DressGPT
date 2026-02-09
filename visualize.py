import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import re

# 1. 讀取資料
df = pd.read_csv('dress_dataset.csv')

# 2. 自動標註函數：從 prompt 提取人種資訊
def get_group(row):
    if row['race'] == 'asian':
        r = 'Asian'
    elif row['race'] == 'caucasian':
        r = 'Caucasian'

    if row['gender'] == 'male':
        g = 'Male'
    elif row['gender'] == 'female':
        g = 'Female'
        
    return f"{r} {g}"

# 執行自動分類
df['group'] = df.apply(get_group, axis=1)

# --- 在 visualize.py 中間加入 ---
# 1. 印出統計摘要，確認分類狀況
print(df['group'].value_counts())

# 2. 使用 groupby 進行聚合計算
group_stats = df.groupby('group')['score'].agg(['mean', 'std', 'count']).sort_values(by='mean', ascending=False)

print("\n" + "="*40)
print("📊 DressGPT 分組統計報告")
print("="*40)
print(group_stats.round(2)) # 四捨五入到小數點後兩位
print("="*40)

# 3. 額外分析：計算各組與理想平均 (5.0) 的差距
ideal_mean = 5.0
for group, row in group_stats.iterrows():
    diff = row['mean'] - ideal_mean
    status = "偏高 ⬆️" if diff > 0 else "偏低 ⬇️"
    print(f"[{group}] 平均分: {row['mean']:.2f} | 與理想差距: {diff:+.2f} ({status})")

# ----------------------------------

# 3. 定義指定顏色對照表 (依照要求設定)
palette_colors = {
    "Asian Female": "pink",
    "Asian Male": "skyblue",
    "Caucasian Female": "orange",
    "Caucasian Male": "lightgreen",
}

# 4. 設定畫布
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# 5. 繪製疊加直方圖 (multiple="stack")
sns.histplot(
    data=df, 
    x='score', 
    hue='group', 
    multiple="stack",     # 核心功能：四段疊加 
    palette=palette_colors, 
    hue_order=["Asian Female", "Asian Male", "Caucasian Female", "Caucasian Male"],
    bins=20, 
    stat="density",
    edgecolor="white",
    alpha=0.8
)

# 6. 繪製理想常態分佈 (mu=5, std=1.5)
x = np.linspace(0, 10, 100)
p = norm.pdf(x, 5, 1.5)
plt.plot(x, p, 'r', linewidth=2, label='Ideal Normal Dist. (mu=5, std=1.5)')

# 7. 標註界線
plt.axvline(10, color='gold', linestyle='--', label='Upper Limit')
plt.axvline(0, color='black', linestyle='--', label='Lower Limit')

# 8. 圖表優化
plt.title('DressGPT Scoring Distribution by Group (Ensemble v8)', fontsize=15)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title="Demographic Groups")

plt.show()



