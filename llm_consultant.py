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

    def generate_advice(self, user_score, analysis_results):
        # 評分標準參考：平均 5.0, 標準差 1.5
        status = "優秀" if user_score > 6.5 else "一般" if user_score > 3.5 else "需改進"
        
        # 這裡建議從 analysis_results 傳入範例的標籤描述 (如果有的話)
        # 若暫無標籤，我們強制 LLM 根據得分分布進行主觀推斷
        prompt = f"""
        你是一位頂級時裝導師。
        
        [評分體系說明]
        - 系統平均分為 5.00，標準差為 1.50。
        - 使用者得分：{user_score:.2f} ({status})。
        
        [數據分析內容]
        - 你的穿搭與此高分範例相似：{analysis_results['like_good_example']}
        - 你的穿搭也帶有此低分範例的影子：{analysis_results['like_bad_example']}
        
        [任務要求]
        1. 專業診斷：根據高分範例，指出使用者目前「已經做對了什麼」（優點）。
        2. 致命傷點評：對照低分範例，直接、有主見地指出「哪個部分毀了整體感」（缺點）。
        3. 指令下達：不要問使用者想改哪裡。直接請他用畫筆塗掉你認為(不是使用者認為)最礙眼的那個部分。
        
        [語氣限制]
        語氣要肯定、自信。使用繁體中文，150字以內。
        """
        try:
            # 新版 SDK 的呼叫方式：models.generate_content
            response = self.client.models.generate_content(
                model=self.model_name, 
                contents=prompt
            )
            return response.text
        except Exception as e:
            # 輔助偵錯：印出完整錯誤細節
            print(f"Gemini API 實際錯誤內容: {e}")
            return f"生成建議時發生錯誤: {str(e)}"