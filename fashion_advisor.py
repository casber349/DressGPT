import torch
import pandas as pd
import re
from torch.nn.functional import cosine_similarity
import json
import os

class FashionAdvisor:
    def __init__(self, db_path='image_embeddings.pt', csv_path='dress_dataset.csv'):
        self.db_dict = torch.load(db_path)
        self.df = pd.read_csv(csv_path)
        self.df['id_str'] = self.df['id'].apply(lambda x: str(x).zfill(4))

        # 載入藥力 (標籤對分數的影響力)
        potency_path = 'labels_potency.json'
        if os.path.exists(potency_path):
            with open(potency_path, 'r', encoding='utf-8') as f:
                self.potency_map = json.load(f)
        else:
            self.potency_map = {}
        
        self.weight_map = {
            0.1: 0.7, 0.2: 0.72, 0.3: 0.75, 0.4: 0.77, 0.5: 0.8,
            0.6: 0.83, 0.7: 0.86, 0.8: 0.9, 0.9: 0.95, 1.0: 1.0,
            1.1: 1.05, 1.2: 1.1, 1.3: 1.13, 1.4: 1.16, 1.5: 1.2,
            1.6: 1.22, 1.7: 1.25, 1.8: 1.27, 1.9: 1.3
        }

        embedding_list = [self.db_dict.get(str_id, torch.zeros((1, 512))) for str_id in self.df['id_str']]
        self.embeddings = torch.cat(embedding_list, dim=0)

    def analyze(self, user_embed, user_tags, user_diagnosis):
        """
        核心升級：回傳完整的診斷對比數據供 LLM 使用
        """
        if user_embed.dim() == 3: user_embed = user_embed.squeeze(0)
        
        gender_mask = (self.df['gender'] == user_tags['gender'])
        pos_mask = (self.df['score'] >= 6.5) & gender_mask
        neg_mask = (self.df['score'] <= 3.5) & gender_mask
        
        similarities = cosine_similarity(user_embed, self.embeddings)
        
        # 1. 取得最強榜樣 (1位)
        pos_sims = similarities * pos_mask.values
        pos_sims[pos_sims == 0] = -1.0
        best_good_idx = torch.argmax(pos_sims).item()
        good_row = self.df.iloc[best_good_idx]
        
        # 2. 取得最慘對象 (1位)
        neg_sims = similarities * neg_mask.values
        neg_sims[neg_sims == 0] = -1.0
        best_neg_idx = torch.argmax(neg_sims).item()
        
        # 3. 整理我的標籤報告 (區分好壞)
        my_good_labels = []
        my_bad_labels = []
        for tag, weight in user_diagnosis:
            potency = self.potency_map.get(tag, 0.0)
            if potency > 0 and weight >= 0.5:
                my_good_labels.append(f"{tag}({weight})")
            elif potency < 0 and weight >= 0.5:
                my_bad_labels.append(f"{tag}({weight})")

        # 4. 解析榜樣的標籤
        good_pos_tags = self._parse_to_list(good_row.get('pos_tags', ""))

        return {
            "good_id": good_row['id_str'],
            "bad_id": self.df.iloc[best_neg_idx]['id_str'],
            "like_good_example": good_row['img_path'],
            "like_bad_example": self.df.iloc[best_neg_idx]['img_path'],
            # --- 餵給 LLM 的關鍵數據 ---
            "user_report": {
                "strengths": my_good_labels,  # 我優秀的標籤
                "weaknesses": my_bad_labels   # 我帶毒的標籤
            },
            "neighbor_report": {
                "good_tags": good_pos_tags      # 榜樣值得學習的標籤
            }
        }

    def get_precision_prescription(self, user_diagnosis, good_id):
        """
        方案 2 實作：只向一個高分榜樣學習，避免風格打架
        """
        u_dict = {tag: weight for tag, weight in user_diagnosis}
        good_row = self.df[self.df['id_str'] == good_id].iloc[0]
        
        # 解析單一榜樣標籤
        n_tags = re.findall(r'\(?([^:,\(\)\s]+):([\d\.]+)\)?', str(good_row.get('pos_tags', "")))
        
        healing_candidates = []
        for tag, n_w_str in n_tags:
            n_w = float(n_w_str)
            u_w = u_dict.get(tag, 0.0)
            delta = max(0, n_w - u_w) # 劑量差
            potency = self.potency_map.get(tag, 0.0)
            if potency > 0:
                healing_candidates.append((tag, n_w, delta * potency))

        # 負向清除
        killing_candidates = []
        for tag, u_w in u_dict.items():
            potency = self.potency_map.get(tag, 0.0)
            if potency < 0:
                killing_candidates.append((tag, u_w, u_w * abs(potency)))

        healing_candidates.sort(key=lambda x: x[2], reverse=True)
        killing_candidates.sort(key=lambda x: x[2], reverse=True)

        rx_pos = [f"({tag}:{self.weight_map.get(round(w, 1), 1.1)})" for tag, w, _ in healing_candidates[:3]]
        rx_neg = [f"({tag}:{self.weight_map.get(round(w + 0.2, 1), 1.2)})" for tag, w, _ in killing_candidates[:3]]

        return ", ".join(rx_pos), ", ".join(rx_neg)

    def _parse_to_list(self, tag_str):
        if pd.isna(tag_str) or tag_str == "": return []
        return [m[0].strip() for m in re.findall(r'\(?([^:,\(\)\s]+):([\d\.]+)\)?', tag_str)]

    def get_inpaint_configs(self, analysis_results, user_tags, user_diagnosis):
        rx_pos, rx_neg = self.get_precision_prescription(user_diagnosis, analysis_results['good_id'])
        
        base_p = "(masterpiece, high quality:1.2), (accurate body features, full body photo:1.15)"
        base_n = "(low quality, gaps:1.15), (extra limbs, missing limbs, bad anatomy, deformed:1.1)"
        
        gender_word = "handsome man" if user_tags['gender'] == 'male' else "beautiful woman"
        cond_p = f"({gender_word}:1.1), ({user_tags['season']} outfit:1.05)"
        
        return f"{base_p}, {cond_p}, {rx_pos}", f"{base_n}, {rx_neg}"