import torch
import pandas as pd
from torch.nn.functional import cosine_similarity

class FashionAdvisor:
    def __init__(self, db_path='image_embeddings.pt', csv_path='dress_dataset.csv'):
        # 1. 載入原始字典資料 { "0001": tensor, "0002": tensor, ... }
        self.db_dict = torch.load(db_path)
        self.df = pd.read_csv(csv_path)
        
        # 2. 依照 CSV 裡的 ID 順序提取 Embedding，確保索引與 CSV 對齊
        # 假設 CSV 的 id 欄位是整數或字串，需補零至 4 位以匹配字典 Key
        embedding_list = []
        for img_id in self.df['id']:
            str_id = str(img_id).zfill(4)
            if str_id in self.db_dict:
                embedding_list.append(self.db_dict[str_id])
            else:
                # 若找不到則補零向量避免長度不一
                embedding_list.append(torch.zeros((1, 512)))
        
        # 3. 將 list 堆疊成 [N, 512] 的大張量
        self.embeddings = torch.cat(embedding_list, dim=0)
        print(f"✅ 成功載入 {self.embeddings.shape[0]} 筆向量資料")

    def analyze(self, user_embed):
        # 確保 user_embed 的維度是 [1, 512]
        if user_embed.dim() == 3: # 有時 clip 會多出一維
            user_embed = user_embed.squeeze(0)

        # 1. 定義高分與低分閾值
        pos_mask = self.df['score'] >= 7.0
        neg_mask = self.df['score'] <= 3.0
        
        # 2. 計算相似度
        # user_embed shape: [1, 512], self.embeddings shape: [N, 512]
        similarities = cosine_similarity(user_embed, self.embeddings)
        
        # 3. 找出最相似的高分樣本 (榜樣)
        pos_sims = similarities * pos_mask.values
        best_pos_idx = torch.argmax(pos_sims)
        best_pos_score = pos_sims[best_pos_idx].item()
        
        # 4. 找出最相似的低分樣本 (地雷)
        neg_sims = similarities * neg_mask.values
        best_neg_idx = torch.argmax(neg_sims)
        best_neg_score = neg_sims[best_neg_idx].item()

        return {
            "like_good_example": self.df.iloc[best_pos_idx.item()]['img_path'],
            "like_bad_example": self.df.iloc[best_neg_idx.item()]['img_path'],
            "pos_similarity": best_pos_score,
            "neg_similarity": best_neg_score
        }
    
    def get_inpaint_configs(self, analysis_results, user_tags):
        """
        根據分析結果與使用者標籤，產生重繪用的 Prompt 與 Negative Prompt
        """
        # 取得榜樣 ID (例如: 0025.jpg)
        good_id = analysis_results.get('like_good_example', "")
        
        # --- [第 5 階段預留位置] ---
        # 等你標記完 CSV 後，這裡會改成從 df 讀取 pos_tags 和 neg_tags
        # 現在我們先用基礎的邏輯確保它能跑
        
        # 基礎正向提示 (確保畫質)
        positive_base = "high quality, professional fashion photography, realistic fabric texture, masterpiece"
        
        # 根據場合給予風格描述
        style_context = user_tags.get('formal', 'casual')
        gender = user_tags.get('gender', 'person')
        
        # 臨時的動態邏輯：如果是特定的榜樣(0025)，給予特殊關鍵字
        custom_hint = ""
        # if good_id == "0025.jpg":
        #     custom_hint = ", minimalist aesthetic, clean tailored lines"
        
        # 組合 Prompt
        prompt = f"{positive_base}, a {gender} in {style_context} outfit{custom_hint}, highly detailed"
        
        # 基礎負向提示 (避免崩壞與拼貼感)
        negative_prompt = (
            "lowres, bad anatomy, worst quality, low quality, blurry, deformed, "
            "clashing colors, messy wrinkles, flat texture, cartoon, 2d, sketch"
        )

        return prompt, negative_prompt