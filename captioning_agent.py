import json
import base64
import io
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

class ImageCaptioningAgent:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        print(f"Initializing model and processor: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def load_image(self, image_input):
        """Loads image from file path or base64 string."""
        if image_input.startswith("data:image") or len(image_input) > 2000:
            # Assume base64
            if "," in image_input:
                header, image_input = image_input.split(",", 1)
            image_bytes = base64.b64decode(image_input)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            # Assume file path
            return Image.open(image_input).convert("RGB")

    def get_caption(self, image_input):
        try:
            image = self.load_image(image_input)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate caption with more detail
            out = self.model.generate(
                **inputs, 
                max_new_tokens=100,
                min_new_tokens=20,
                num_beams=5,
                early_stopping=True
            )
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return {"caption": caption}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Image Captioning Agent")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--base64", type=str, help="Base64 encoded image string")
    args = parser.parse_args()

    agent = ImageCaptioningAgent()

    if args.image:
        result = agent.get_caption(args.image)
    elif args.base64:
        result = agent.get_caption(args.base64)
    else:
        print("Please provide either --image or --base64")
        return

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
