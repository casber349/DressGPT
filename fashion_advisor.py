import torch
import pandas as pd
import re
from torch.nn.functional import cosine_similarity
import json
import os

from feature_utils import get_one_hot_tags # 引入你的工具

class FashionAdvisor:
    def __init__(self, db_path='image_embeddings.pt', csv_path='dress_dataset.csv'):
        self.db_dict = torch.load(db_path)
        self.df = pd.read_csv(csv_path)
        self.df['id_str'] = self.df['id'].apply(lambda x: str(x).zfill(4))

        # 💡 新增：預先算好資料庫的 15 維條件，方便快速過濾
        cond_list = []
        for _, row in self.df.iterrows():
            cond_list.append(get_one_hot_tags(row))
        self.cond_matrix = torch.stack(cond_list) # [N, 15]

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

    def _get_weight_template(self, user_tags):
        # 構建「邏輯相似度」權重向量 (15維)
        # 根據你提供的對照表，建立一個與 user_tags 相對應的權重模板
        weight_template = torch.zeros(15)
        
        # --- A. 性別 (0-1) ---
        u_g = user_tags.get('gender', 'male')
        if u_g == 'male': weight_template[0:2] = torch.tensor([1.0, 0.0])
        else:             weight_template[0:2] = torch.tensor([0.0, 1.0])

        # --- B. 年齡 (2-5) ---
        u_a = user_tags.get('age', 'adult')
        if u_a == 'teenager':    weight_template[2:6] = torch.tensor([1.0, 0.8, 0.0, 0.0])
        elif u_a == 'adult':     weight_template[2:6] = torch.tensor([0.6, 1.0, 0.7, 0.0])
        elif u_a == 'middle-aged': weight_template[2:6] = torch.tensor([0.0, 0.6, 1.0, 0.2])
        else:                    weight_template[2:6] = torch.tensor([0.0, 0.0, 0.5, 1.0])

        # --- C. 身材 (6-9) ---
        u_b = user_tags.get('body', 'average')
        if u_b == 'average':   weight_template[6:10] = torch.tensor([1.0, 0.4, 0.4, 0.0])
        elif u_b == 'skinny':  weight_template[6:10] = torch.tensor([0.6, 1.0, 0.0, 0.0])
        elif u_b == 'athletic': weight_template[6:10] = torch.tensor([0.5, 0.0, 1.0, 0.3])
        else:                  weight_template[6:10] = torch.tensor([0.0, 0.0, 0.2, 1.0])

        # --- D. 季節 (10-12) ---
        u_s = user_tags.get('season', 'spring/fall')
        if u_s == 'summer':      weight_template[10:13] = torch.tensor([1.0, 0.0, 0.4])
        elif u_s == 'winter':    weight_template[10:13] = torch.tensor([0.0, 1.0, 0.2])
        else:                    weight_template[10:13] = torch.tensor([0.5, 0.6, 1.0])

        # --- E. 正式度 (13-14) ---
        u_f = user_tags.get('formal', 'casual')
        if u_f == 'casual':      weight_template[13:15] = torch.tensor([1.0, 0.6])
        else:                    weight_template[13:15] = torch.tensor([0.0, 1.0])

        return weight_template

    def _calculate_logic_scores(self, user_tags):
        # 取得權重模板 (w 是 15 維)
        w = self._get_weight_template(user_tags)
        
        # A. 初步檢測 (五項連乘)：只要其中一項是 0，結果就是 0 (一票否決)
        # 這裡是用來過濾掉「性別不符」或「絕對不搭」的樣本
        s_gen = (self.cond_matrix[:, 0:2] * w[0:2]).sum(dim=1)
        s_age = (self.cond_matrix[:, 2:6] * w[2:6]).sum(dim=1)
        s_bod = (self.cond_matrix[:, 6:10] * w[6:10]).sum(dim=1)
        s_sea = (self.cond_matrix[:, 10:13] * w[10:13]).sum(dim=1)
        s_for = (self.cond_matrix[:, 13:15] * w[13:15]).sum(dim=1)
        
        # 建立 Pass Filter (只有全不為 0 的才是 1.0，其餘為 0.0)
        # 用於確保「硬過濾」生效
        pass_filter = (s_gen > 0) & (s_age > 0) & (s_bod > 0) & (s_sea > 0) & (s_for > 0)
        pass_filter = pass_filter.float()

        # B. 條件分數計算 (加法處理)：將五項分數相加，滿分 5.0
        # 這能保留「軟過濾」的彈性 (例如 0.8 分的年齡權重)
        total_condition_score = s_gen + s_age + s_bod + s_sea + s_for
        
        # C. 最終正規化邏輯分 = (條件總分 / 5.0) * 生殺過濾器
        # 這樣當過關時，分數會在 0.2 ~ 1.0 之間；不選中時必為 0.0
        normalized_logic_scores = (total_condition_score / 5.0) * pass_filter
        
        return normalized_logic_scores
    
    def _find_best_idx(self, final_scores, logic_scores, priority, fallback, mode, min_match=0.4):
        db_scores = torch.tensor(self.df['score'].values, device=final_scores.device)
        
        # 定義搜尋階段
        stages = [
            {"threshold": priority, "strict": True},  # 第一門檻：精英模式 (需滿足 min_match)
            {"threshold": fallback, "strict": False}  # 第二門檻：生存模式 (只要 Logic > 0)
        ]

        for stage in stages:
            threshold = stage["threshold"]
            if threshold is None: continue
            
            # 建立遮罩
            score_mask = (db_scores >= threshold) if mode == 'high' else (db_scores <= threshold)
            full_mask = score_mask & (logic_scores > 0.0)
            
            temp_cands = final_scores.clone()
            temp_cands[~full_mask] = -999.0
            idx = torch.argmax(temp_cands).item()
            
            match_val = temp_cands[idx].item()
            
            # 判定邏輯
            if match_val > -500:
                # 如果是嚴格模式，必須超過 min_match
                if stage["strict"] and match_val < min_match:
                    print(f"DEBUG: {mode} 優先對象相似度不足({match_val:.3f} < {min_match})，切換至備案...")
                    continue 
                
                # 輸出 Debug 資訊 (此處僅輸出至終端機)
                print(f"--- 找到 {mode} 對象 ---")
                print(f"ID: {self.df.iloc[idx]['id_str']}, Match: {match_val:.4f}, Mode: {'Strict' if stage['strict'] else 'Fallback'}")
                return idx
                
        return None

    def analyze(self, user_embed, user_tags, user_diagnosis, user_score):
        if user_embed.dim() == 3: user_embed = user_embed.squeeze(0)
        
        # 1. 視覺相似度
        vis_sims = cosine_similarity(user_embed, self.embeddings)

        # 2. 邏輯權重分 (包含 0.0 一票否決)
        logic_scores = self._calculate_logic_scores(user_tags)

        # 3. 融合分數 (相似度 * 條件權重)
        final_scores = vis_sims * logic_scores

        # --- 4. 根據你的策略設定動態門檻 ---
        if user_score >= 6.50:
            g_pri, g_fall = user_score + 0.10, 6.50
            n_pri, n_fall = 5.00, None
        elif 3.50 <= user_score < 6.50:
            g_pri, g_fall = 6.50, user_score + 0.10
            n_pri, n_fall = 3.50, user_score - 0.10
        else: # < 3.50
            g_pri, g_fall = 5.00, None
            n_pri, n_fall = user_score - 0.10, 3.50

        # --- 5. 執行搜尋 ---
        # 尋找好鄰居
        best_good_idx = self._find_best_idx(final_scores, logic_scores, g_pri, g_fall, mode='high')
        if best_good_idx is None:
            return None # 告訴 app.py 徹底找不到人

        # 尋找壞鄰居 (找不到就維持 None)
        best_neg_idx = self._find_best_idx(final_scores, logic_scores, n_pri, n_fall, mode='low')
        if best_neg_idx is None:
            return None # 告訴 app.py 徹底找不到人

        # --- 6. 後續處理與回傳 ---
        good_row = self.df.iloc[best_good_idx]
        
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