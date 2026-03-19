import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import json
import os
from utils import load_image, pil_to_base64

class InpaintingSpecialist:
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Initializing InpaintingSpecialist on {self.device}...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype
        )
        self.pipe = self.pipe.to(self.device)

    def run(self, image_input, mask_input, prompt, negative_prompt=None, guidance_scale=7.5, strength=1.0):
        """
        Runs the inpainting pipeline.
        :param image_input: File path or base64 string of the original image.
        :param mask_input: File path or base64 string of the binary mask.
        :param prompt: Text description for the masked area.
        :param negative_prompt: What should NOT be in the result.
        :param guidance_scale: How much to follow the prompt.
        :param strength: How much to transform the masked area (0.0 to 1.0).
        :return: JSON string with the inpainted image as base64.
        """
        # 1. Load images
        image = load_image(image_input).convert("RGB")
        mask = load_image(mask_input).convert("L")

        # 2. Mask Preprocessing: Feathering/Blurring
        # This helps blend the inpainted area with the rest of the image
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.GaussianBlur(radius=4))

        # 3. Run inference
        steps = 20 if self.device == "cpu" else 50
        
        # Default negative prompt to help preserve character if they were partially masked
        if negative_prompt is None:
            negative_prompt = "deformed face, ugly, blurry, low quality, distorted features, changed person"

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            strength=strength
        ).images[0]

        # 4. Convert to base64
        inpainted_base64 = pil_to_base64(result)

        # 5. Return JSON
        return json.dumps({
            "inpainted_image": inpainted_base64
        })

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        img_path = sys.argv[1]
        mask_path = sys.argv[2]
        prompt_text = " ".join(sys.argv[3:])
        
        specialist = InpaintingSpecialist()
        try:
            output_json = specialist.run(img_path, mask_path, prompt_text)
            print(output_json)
        except Exception as e:
            print(json.dumps({"error": str(e)}))
    else:
        print("Usage: python inpainting_specialist.py <image_path/base64> <mask_path/base64> <prompt>")
