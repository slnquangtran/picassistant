import json
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from utils import load_image, pil_to_base64, calculate_iou

class SegmentationAgent:
    def __init__(self, model_name="facebook/sam-vit-base"):
        """
        Initializes the SAM segmentation pipeline.
        """
        print(f"Initializing SegmentationAgent with {model_name}...")
        self.segmenter = pipeline("mask-generation", model=model_name, device=0 if torch.cuda.is_available() else -1)

    def run(self, image_input, bboxes=None):
        """
        Runs segmentation on the input image using SAM.
        If bboxes are provided, generates masks restricted to those boxes.
        """
        image = load_image(image_input).convert("RGB")
        
        if bboxes:
            # SAM pipeline expects bboxes in [xmin, ymin, xmax, ymax]
            # formatted_bboxes: [{"label": "...", "box": [xmin, ymin, xmax, ymax], ...}]
            sam_boxes = [d['box'] for d in bboxes]
            
            # Run SAM with boxes
            outputs = self.segmenter(image, input_boxes=[sam_boxes])
            # For a single image, outputs is a dict: {"masks": ..., "scores": ...}
            masks = outputs["masks"] 
            
            # If it's a list (typical for some transformers versions), it contains masks
            filtered_results = []
            if isinstance(masks, list):
                # masks[i] corresponds to box[i]
                for mask_entry in masks:
                    # sometimes SAM returns 3 masks per box (multimask_output)
                    # mask_entry might be a list of 3 masks or a single mask
                    if isinstance(mask_entry, list):
                         mask_to_use = mask_entry[0]
                    else:
                         mask_to_use = mask_entry
                    
                    if isinstance(mask_to_use, torch.Tensor):
                        mask_to_use = mask_to_use.cpu().numpy()
                    
                    if isinstance(mask_to_use, np.ndarray):
                         # Handle boolean or uint8
                         if mask_to_use.dtype == bool:
                             mask_to_use = (mask_to_use * 255).astype('uint8')
                         filtered_results.append(Image.fromarray(mask_to_use))
                    else:
                         filtered_results.append(mask_to_use)
            else:
                # Handle tensor/array case
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                
                # If [num_boxes, 3, H, W]
                if len(masks.shape) == 4:
                     masks = masks[:, 0, :, :]
                elif len(masks.shape) == 5:
                     masks = masks[0, :, 0, :, :]
                
                for mask_data in masks:
                    byte_mask = (mask_data * 255).astype('uint8') if mask_data.dtype == bool else mask_data
                    filtered_results.append(Image.fromarray(byte_mask))
        else:
            # Fallback for no bboxes
            outputs = self.segmenter(image)
            masks = outputs[0]["masks"] if isinstance(outputs, list) else outputs["masks"]
            filtered_results = [m if isinstance(m, Image.Image) else Image.fromarray((m*255).astype('uint8')) for m in (masks[0] if isinstance(masks[0], list) else masks)]

        # Convert masks to base64
        base64_masks = [pil_to_base64(mask) for mask in filtered_results]

        return json.dumps({"masks": base64_masks})

if __name__ == "__main__":
    # Example usage (commented out)
    # agent = SegmentationAgent()
    # print(agent.run("path/to/image.jpg"))
    pass
