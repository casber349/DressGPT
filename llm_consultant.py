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

    def generate_advice(self, user_score, analysis_results, is_inpainted=False):
        """使用 Gemini API 生成優美的穿搭評論"""
        mode_text = "【局部修改後的模擬評估】" if is_inpainted else "【原始穿搭診斷】"
        score_status, tone = self._get_status_context(user_score)
        
        # 從分析結果中提取鄰居標籤
        good_tags = analysis_results.get('good_tags', "未提供")
        bad_tags = analysis_results.get('bad_tags', "未提供")

        prompt = f"""
        你是一位具備尖銳審美眼光且懂統計學的 AI 時尚顧問。
        
        [當前任務：{mode_text}]
        - 使用者得分：{user_score:.2f} / 10
        - 統計學地位：{score_status}
        
        [背景數據 - 禁止瞎扯]
        你的診斷必須嚴格基於以下數據。若數據中沒提到的配件或細節，嚴禁憑空想像：
        1. 使用者做得好的「亮點標籤」(參考 ID {analysis_results.get('good_id', 'N/A')})：{good_tags}
        2. 使用者應避開的「雷區標籤」(參考 ID {analysis_results.get('bad_id', 'N/A')})：{bad_tags}
        
        [評論方針]
        - 語氣風格：{tone}
        - 6.5分代表 +1 標準差(PR84)，已經相當優秀。
        - 8分代表 +2 標準差(PR97)，已經是頂尖水準。
        - 5分代表 PR50 (平均水準)，對 3.5 ~ 5 分的人應保持鼓勵。
        - 3.5 分 (-1 標準差) 以下才需要直接指出問題。
        
        [輸出格式]
        1. 穿搭地位 (總結目前 PR 水準)
        2. 亮點分析 (根據亮點標籤進行分析)
        3. 避雷建議 (根據雷區標籤，提供精準的優化建議)
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text

    def generate_backup_advice(self, user_score, analysis_results):
        """備用函數：當 API 不可用時，直接將標籤數據格式化輸出"""
        score_status, _ = self._get_status_context(user_score)
        good_tags = analysis_results.get('good_tags', "暫無數據")
        bad_tags = analysis_results.get('bad_tags', "暫無數據")

        # 返回 HTML 格式，讓 App 介面直接渲染
        backup_html = f"""
        <div class="backup-advice" style="border-left: 4px solid #3498db; padding-left: 15px;">
            <p style="color: #666; font-size: 0.9em;">(⚠️ AI 顧問目前連線不穩定，以下為原始分析數據)</p>
            <p><strong>統計地位：</strong> {score_status} ({user_score:.2f}分)</p>
            <p><strong>💡 亮點參考：</strong><br><small>{good_tags}</small></p>
            <p><strong>⚠️ 避雷參考：</strong><br><small>{bad_tags}</small></p>
        </div>
        """
        return backup_html