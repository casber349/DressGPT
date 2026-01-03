import os
from google import genai # 注意這裡的匯入方式改變了

class DressConsultant:
    def __init__(self):
        # 從環境變數中讀取 API Key
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("❌ 錯誤：找不到 GEMINI_API_KEY。")
            
        # 使用新版 SDK 的 Client 初始化
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.5-flash'

    def generate_advice(self, user_score, analysis_results, is_inpainted=False):
        # 根據是否已經過 Inpaint，調整診斷情境
        mode_text = "【局部修改後的模擬評估】" if is_inpainted else "【原始穿搭診斷】"
        
        # 根據分數分佈判斷強弱 (平均 5.0, 標準差 1.5)
        if user_score > 6.5:
            score_desc = "表現優異，接近時尚指標。"
        elif user_score < 3.5:
            score_desc = "有明顯的改進空間，目前造型顯得雜亂。"
        else:
            score_desc = "表現平平，中規中矩但缺乏亮點。"

        prompt = f"""
        你是一位專業且具備尖銳審美眼光的 AI 時尚顧問。
        
        [當前任務：{mode_text}]
        - 使用者當前穿搭得分：{user_score:.2f} / 10
        - 評分標準參考：平均值 5.00，標準差 1.50。({score_desc})
        
        [數據分析對比]
        1. 優點：使用者與高分範例 (ID: {analysis_results['like_good_example']}) 有相似的優點。請指出其在比例、色系或風格上的成功之處。
        2. 缺點：使用者與低分範例 (ID: {analysis_results['like_bad_example']}) 有相似的失誤。請直接、主見地指出哪個細節「毀了整體感」。
        
        [任務要求]
        {"這是一張模擬修改後的結果，請評估這次修改是否成功提升了質感？" if is_inpainted else "請引導使用者『手畫遮罩 (Mask)』來塗掉你認為最不理想的部分，讓我們進行 AI 模擬換裝。"}
        
        語氣要求：專業、自信。
        字數限制：繁體中文回答，150 字以內。
        """
        try:
            # 新版 SDK 的呼叫方式：models.generate_content
            response = self.client.models.generate_content(
                model=self.model_name, 
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Gemini API 實際錯誤內容: {e}")
            # 如果是額度滿了 (429)，回傳更直觀的提示
            if "429" in str(e) or "quota" in str(e).lower():
                return "【系統提示】AI 顧問目前因額度限制暫時離線，但您可以參考分數與對照圖進行調整。"
            return "生成建議時發生異常，請稍後再試。"