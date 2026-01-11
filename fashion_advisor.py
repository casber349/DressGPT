import torch
import pandas as pd
import re
from torch.nn.functional import cosine_similarity

class FashionAdvisor:
    def __init__(self, db_path='image_embeddings.pt', csv_path='dress_dataset.csv'):
        self.db_dict = torch.load(db_path)
        self.df = pd.read_csv(csv_path)
        self.df['id_str'] = self.df['id'].apply(lambda x: str(x).zfill(4))
        
        # 權重映射表 (你的 19 種精準對應)
        self.weight_map = {
            0.1: 0.7, 0.2: 0.72, 0.3: 0.75, 0.4: 0.77, 0.5: 0.8,
            0.6: 0.83, 0.7: 0.86, 0.8: 0.9, 0.9: 0.95, 1.0: 1.0,
            1.1: 1.05, 1.2: 1.1, 1.3: 1.13, 1.4: 1.16, 1.5: 1.2,
            1.6: 1.22, 1.7: 1.25, 1.8: 1.27, 1.9: 1.3
        }

        # 預載 Embedding
        embedding_list = [self.db_dict.get(str_id, torch.zeros((1, 512))) for str_id in self.df['id_str']]
        self.embeddings = torch.cat(embedding_list, dim=0)

    def analyze(self, user_embed, user_tags):
        """
        條件過濾邏輯：只在符合性別的樣本中找鄰居
        """
        if user_embed.dim() == 3: user_embed = user_embed.squeeze(0)
        
        # 1. 硬性過濾：性別必須一致
        gender_mask = (self.df['gender'] == user_tags['gender'])
        
        # 2. 門檻設定 (6.5/3.5)
        pos_mask = (self.df['score'] >= 6.5) & gender_mask
        neg_mask = (self.df['score'] <= 3.5) & gender_mask
        
        similarities = cosine_similarity(user_embed, self.embeddings)
        
        # 找出最像的好鄰居與壞鄰居
        best_pos_idx = torch.argmax(similarities * pos_mask.values).item()
        best_neg_idx = torch.argmax(similarities * neg_mask.values).item()

        return {
            "good_id": self.df.iloc[best_pos_idx]['id_str'],
            "bad_id": self.df.iloc[best_neg_idx]['id_str'],
            "like_good_example": self.df.iloc[best_pos_idx]['img_path'],
            "like_bad_example": self.df.iloc[best_neg_idx]['img_path']
        }

    def _parse_tags(self, tag_str):
        """解析標籤並套用權重映射"""
        if pd.isna(tag_str) or tag_str == "": return ""
        matches = re.findall(r'\(([^:]+):([\d\.]+)\)', tag_str)
        return ", ".join([f"({m[0].strip()}:{self.weight_map.get(round(float(m[1]), 1), 1.0)})" for m in matches])

    def get_inpaint_configs(self, analysis_results, user_tags):
        # 1. 取得鄰居原始標籤 (好鄰居取 pos_tags, 壞鄰居取 neg_tags)
        good_row = self.df[self.df['id_str'] == analysis_results['good_id']].iloc[0]
        bad_row = self.df[self.df['id_str'] == analysis_results['bad_id']].iloc[0]
        
        neighbor_pos = self._parse_tags(good_row.get('pos_tags', ""))
        neighbor_neg = self._parse_tags(bad_row.get('neg_tags', ""))

        # 2. 組合三段式 Prompt
        # A. 基礎標籤
        base_p = "(masterpiece, high quality:1.2), (accurate body features:1.15), (full body photo:1.15)"
        base_n = "(low quality, gaps:1.15), (extra limbs, missing limbs:1.1), (bad anatomy, deformed:1.1)"

        # B. 條件過濾標籤
        gender_word = "man" if user_tags['gender'] == 'male' else "woman"
        cond_p = f"({gender_word} standing:1.1), ({user_tags['season']} outfit:1.05)"
        if user_tags['formal'] == 'formal': cond_p += ", (formal:1.1)"
        
        # C. 負向條件過濾 (防止跑出異性特徵)
        cond_n = "dress, skirt, bra, feminine" if user_tags['gender'] == 'male' else "masculine"

        # 3. 最終拼接
        final_prompt = f"{base_p}, {cond_p}, {neighbor_pos}"
        final_neg = f"{base_n}, {cond_n}, {neighbor_neg}"

        return final_prompt, final_neg