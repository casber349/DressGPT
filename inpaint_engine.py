import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import os

class InpaintEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "runwayml/stable-diffusion-inpainting"
        
        print(f"🚀 正在載入 Inpaint 模型至 {self.device}...")
        
        # 使用 float16 節省一半顯存
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None # 關閉安全檢查以加速並避免誤判
        ).to(self.device)

        # 顯存優化：如果顯存低於 8GB，啟用以下設定
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            # self.pipe.enable_model_cpu_offload() # 如果還是炸顯存再開這行

    def generate(self, image_path, mask_path, prompt, negative_prompt):
        # 讀取圖片
        init_image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")
        
        # 這裡確保縮放回 SD 較好處理的尺寸，例如 (576, 1024) 
        # 或是如果你顯存較小，可以縮放成 (448, 800)
        init_image = init_image.resize((576, 1024))
        mask_image = mask_image.resize((576, 1024))

        with torch.autocast("cuda"):
            # 執行重繪
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=30,
                strength=1.0,
                guidance_scale=7.0,
                padding_mask_crop=32
            ).images[0]
        
        return output.resize((576, 1024))