import json
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from utils import load_image, pil_to_base64, calculate_iou

class SegmentationAgent:
    def __init__(self, model_name="facebook/sam-vit-large"):
        """
        Initializes the SAM segmentation pipeline with a higher capacity model.
        """
        print(f"Initializing SegmentationAgent with {model_name}...")
        device = 0 if torch.cuda.is_available() else -1
        self.segmenter = pipeline("mask-generation", model=model_name, device=device)

    def run(self, image_input, bboxes=None, points=None):
        """
        Runs segmentation on the input image using SAM.
        Includes post-processing for smoother masks.
        """
        from PIL import ImageFilter
        
        image = load_image(image_input).convert("RGB")
        
        # Determine the prompt type
        if bboxes:
            sam_boxes = [d['box'] for d in bboxes]
            outputs = self.segmenter(image, input_boxes=[sam_boxes])
        elif points:
            outputs = self.segmenter(image, input_points=[points])
        else:
            outputs = self.segmenter(image)

        # Process outputs
        # SAM can return multiple masks per prompt. We want the best one.
        masks = outputs[0]["masks"] if isinstance(outputs, list) else (outputs["masks"] if "masks" in outputs else outputs)
        
        if isinstance(masks, list) and len(masks) > 0 and isinstance(masks[0], list):
            # Take the first (best) mask for each prompt
            masks = [m_list[0] for m_list in masks]
        elif not isinstance(masks, list):
            masks = [m for m in masks]

        processed_masks = []
        for m in masks:
            if isinstance(m, dict) and "segmentation" in m:
                m = m["segmentation"]
            
            # Convert to numpy boolean if needed
            if not isinstance(m, Image.Image):
                arr = m.cpu().numpy() if isinstance(m, torch.Tensor) else m
                byte_mask = (arr * 255).astype('uint8') if arr.dtype == bool else arr
                mask_img = Image.fromarray(byte_mask)
            else:
                mask_img = m.convert("L")

            # --- POST-PROCESSING: Smoothing ---
            # 1. Apply a bit of Gaussian Blur to soften edges
            # 2. Threshold back to binary
            # This reduces "staircase" effects on tilted lines
            smoothed = mask_img.filter(ImageFilter.GaussianBlur(radius=1))
            binary = smoothed.point(lambda p: 255 if p > 128 else 0)
            
            processed_masks.append(binary)

        base64_masks = [pil_to_base64(mask) for mask in processed_masks]
        return json.dumps({"masks": base64_masks})

        # Convert masks to base64
        base64_masks = [pil_to_base64(mask) for mask in filtered_results]

        return json.dumps({"masks": base64_masks})

if __name__ == "__main__":
    # Example usage (commented out)
    # agent = SegmentationAgent()
    # print(agent.run("path/to/image.jpg"))
    pass
