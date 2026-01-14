import os
from google import genai

class DressConsultant:
    def __init__(self):
        # 從環境變數中讀取 API Key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ 錯誤：找不到 GEMINI_API_KEY。")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.5-flash'

    def _get_status_context(self, user_score):
        """內部工具函數：根據分數定義統計學地位與語氣"""
        if user_score >= 6.5:
            return "極其優秀 (PR84+)，具備模特兒或時尚博主水準。", "以讚美為主，給予細節上的微調建議。"
        elif user_score >= 5.0:
            return "水準以上 (PR50-84)，穿搭體面、符合大眾審美。", "給予肯定，並鼓勵嘗試進階風格。"
        elif user_score >= 3.5:
            return "普通但有提升空間 (PR16-50)，沒有大錯但缺乏亮點。", "平實中肯，指出 1-2 個可以優化的小地方。"
        else:
            return "需大幅改進 (PR16 以下)，穿搭存在明顯的結構性問題。", "直接且具批判性，必須指出毀掉整體感的致命傷。"

    # llm_consultant.py 核心修改
    def generate_advice(self, user_score, analysis_results, is_inpainted=False):
        """
        方案 1 實作：結合模型評分(主觀)與標籤報告(客觀)進行深度評論
        """
        mode_text = "【局部修改後的模擬評估】" if is_inpainted else "【原始穿搭診斷】"
        score_status, tone = self._get_status_context(user_score)
        
        # 提取我們在 advisor 整理好的結構化報告
        user_report = analysis_results.get('user_report', {})
        neighbor_report = analysis_results.get('neighbor_report', {})
        original_score = analysis_results.get('original_score', user_score)

        # 格式化標籤數據供 LLM 閱讀
        my_strengths = ", ".join(user_report.get('strengths', ["尚未偵測到顯著優點"]))
        my_weaknesses = ", ".join(user_report.get('weaknesses', ["尚未偵測到顯著缺點"]))
        targets = ", ".join(neighbor_report.get('good_tags', ["無參考建議"]))

        prompt = f"""
        你是一位具備犀利審美眼光且精通大數據分析的 AI 時尚顧問。
        
        [診斷模式：{mode_text}]
        - 當前得分：{user_score:.2f} / 10
        - 原始分數：{original_score:.2f} / 10 (若分數有變動，請評論進步幅度)
        - 統計學地位：{score_status}
        
        [背景診斷數據 - 嚴禁瞎扯]
        1. 使用者目前的「亮點標籤」(優點)：{my_strengths}
        2. 使用者目前的「雷區標籤」(致命傷)：{my_weaknesses}
        3. 推薦學習榜樣 (ID {analysis_results.get('good_id')}) 的核心優點：{targets}
        
        [評論方針]
        - 語氣風格：{tone}
        - 請結合「主觀評分」與「客觀標籤梯度」進行分析。
        - 如果是重繪後的結果，請重點評論：新加入的標籤是否有效壓制了雷區。
        
        [輸出格式]
        1. **穿搭地位**：一句話總結目前的 PR 水準。
        2. **亮點分析**：針對「亮點標籤」進行評論。
        3. **避雷建議**：針對「雷區標籤」指出為何導致分數下降，並根據「榜樣優點」提出具體改善方向。
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text

    def generate_backup_advice(self, user_score, analysis_results):
        """
        專業備用函數：當 Gemini API 耗盡時，直接輸出基於實測數據的診斷報告
        """
        import math
        
        # 1. 計算真實 PR 值 (Sigma=1.5, Mu=5.0)
        def get_real_pr(s):
            mu, sigma = 5.0, 1.5
            return 0.5 * (1 + math.erf((s - mu) / (sigma * math.sqrt(2)))) * 100
        
        pr = get_real_pr(user_score)
        
        # 2. 從傳入的字典中提取標籤數據
        user_report = analysis_results.get('user_report', {})
        neighbor_report = analysis_results.get('neighbor_report', {})
        
        # 我優秀的與帶毒的標籤
        my_good = user_report.get('strengths', "暫無數據")
        my_bad = user_report.get('weaknesses', "暫無數據")
        
        # 好鄰居值得學習的標籤
        neighbor_good = neighbor_report.get('good_tags', "暫無數據")
        # 獲取唯一的好鄰居 ID
        neighbor_id = analysis_results.get('good_id', "未知")

        # 3. 定義四段位階診斷 (基於你的實測公式)
        if user_score <= 3.5:
            level = "【低分段：結構重塑】"
            diag = "圖像基礎較弱，系統已開啟 100% 強度。建議大面積重繪以重新定義結構。"
        elif user_score <= 6.0:
            level = "【中分段：審美優化】"
            diag = f"當前 PR {pr:.1f}%。處於「慢速衰減區」，強度穩定。建議針對特定單品進行修正。"
        elif user_score <= 7.0:
            level = "【高分段：質感衝刺】"
            diag = f"當前 PR {pr:.1f}%。已進入「快速衰減區」。請務必縮小塗抹面積，專注於材質細節。"
        else:
            level = "【逆天段：神之領域】"
            diag = f"當前 PR {pr:.1f}%。已達底圖極限。不建議大面積重繪，僅適合極小範圍的像素拋光。"

        # 4. 格式化輸出
        advice_lines = [
            "-----------------------------------------",
            " (⚠️ API 配額已達上限，切換至數據診斷模式) ",
            "-----------------------------------------",
            f"{level}",
            f"📊 統計地位：PR {pr:.1f} (得分: {user_score:.2f})",
            f"🩺 專家診斷：{diag}",
            "",
            f"✅ 你的亮點標籤：{my_good}",
            f"❌ 建議改進標籤：{my_bad}",
            f"🌟 榜樣標籤參考 (ID: {neighbor_id})：{neighbor_good}",
            "",
            "💡 提示：高分區重繪應採取「小面積、低強度」策略，避免結構崩壞。",
            "-----------------------------------------"
        ]

        return "\n".join(advice_lines)