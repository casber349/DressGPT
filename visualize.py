import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

# 讀取資料
df = pd.read_csv('dress_dataset.csv')

# 設定畫布
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# 繪製直方圖
sns.histplot(df['score'], kde=False, color='skyblue', stat="density", bins=20, label='Actual Scores')

# 計算目前資料的平均值與標準差
mu, std = df['score'].mean(), df['score'].std()

# 繪製理想的常態分佈曲線 (用於對比)
xmin, xmax = plt.xlim()
x = np.linspace(0, 10, 100)
p = norm.pdf(x, 5, 1.5)
plt.plot(x, p, 'r', linewidth=2, label=f'Normal Dist. (mu=5.00, std=1.50)')

# 標註你的「標竿」
plt.axvline(10, color='gold', linestyle='--', label='Upper Limit')
plt.axvline(0, color='black', linestyle='--', label='Lower Limit')

plt.title('DressGPT Scoring Distribution', fontsize=15)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.show()

print(f"統計摘要：")
print(f"樣本數: {len(df)}")
print(f"平均分 (Mean): {mu:.2f} (目標: 5.00)")
print(f"標準差 (Std): {std:.2f} (目標: 1.50)")