import io
import json
import base64
from PIL import Image
from transformers import pipeline

from transformers import OwlViTProcessor, OwlViTForObjectDetection, pipeline
import torch

# Global models to avoid reloading
OWL_PROCESSOR = None
OWL_MODEL = None

def load_image(image_input):
    """
    Loads an image from a file path or a base64 string.
    """
    if isinstance(image_input, str):
        if image_input.startswith("data:image") or ";base64," in image_input:
            # Handle data URI
            header, encoded = image_input.split(",", 1)
            image_data = base64.b64decode(encoded)
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        try:
            # Check if it's a valid base64 string without header
            image_data = base64.b64decode(image_input)
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception:
            # Treat as file path
            return Image.open(image_input).convert("RGB")
    else:
        raise ValueError("Input must be a file path or base64 string.")

def detect_objects(image_input, queries=None, threshold=0.1):
    """
    Performs object detection using OWL-ViT for open-vocabulary detection.
    """
    global OWL_PROCESSOR, OWL_MODEL
    
    # 1. Load the image
    image = load_image(image_input)
    
    # 2. Initialize the model on first call
    if OWL_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing OWL-ViT on {device}...")
        # Use v2 for better performance if possible, but keeping v1 base for compatibility
        # google/owlvit-base-patch32 is stable
        OWL_PROCESSOR = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        OWL_MODEL = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    
    # 3. Define broad candidate labels if not provided
    if queries is None:
        queries = [
            "person", "girl", "anime character", "dress", "hair", "face", 
            "tree", "forest", "river", "water", "sky", "grass", "leaves",
            "rock", "flower", "mountain", "bridge", "animal", "building", "cloud"
        ]
    elif isinstance(queries, str):
        # Handle comma separated string
        queries = [q.strip() for q in queries.split(",") if q.strip()]
    
    # 4. Run inference
    inputs = OWL_PROCESSOR(text=[queries], images=image, return_tensors="pt").to(OWL_MODEL.device)
    with torch.no_grad():
        outputs = OWL_MODEL(**inputs)
    
    # 5. Post-process
    target_sizes = torch.Tensor([image.size[::-1]])
    results = OWL_PROCESSOR.post_process_grounded_object_detection(
        outputs=outputs, 
        target_sizes=target_sizes, 
        threshold=threshold
    )
    
    detections = []
    i = 0 # Single image batch
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    
    # Sort by score for NMS
    sorted_indices = torch.argsort(scores, descending=True)
    
    seen_boxes = []
    for idx in sorted_indices:
        score = scores[idx].item()
        label_idx = labels[idx].item()
        box = boxes[idx].tolist()
        
        # COCO [xmin, ymin, xmax, ymax] -> [x, y, w, h]
        xmin, ymin, xmax, ymax = box
        w, h = xmax - xmin, ymax - ymin
        
        # More robust IoU based NMS
        current_box = [xmin, ymin, w, h] # Keep as float for precision
        is_duplicate = False
        
        for prev_box in seen_boxes:
            # Intersection
            ixmin = max(current_box[0], prev_box[0])
            iymin = max(current_box[1], prev_box[1])
            ixmax = min(current_box[0] + current_box[2], prev_box[0] + prev_box[2])
            iymax = min(current_box[1] + current_box[3], prev_box[1] + prev_box[3])
            
            iw = max(0, ixmax - ixmin)
            ih = max(0, iymax - iymin)
            intersection = iw * ih
            
            # Union
            area1 = current_box[2] * current_box[3]
            area2 = prev_box[2] * prev_box[3]
            union = area1 + area2 - intersection
            
            iou = intersection / union if union > 0 else 0
            
            # If same label and high IoU, or different label but very high IoU (likely redundant)
            if iou > 0.5:
                is_duplicate = True
                break
        
        if not is_duplicate:
            detections.append({
                "label": queries[label_idx],
                "confidence": round(score, 4),
                "bbox": [int(xmin), int(ymin), int(w), int(h)]
            })
            seen_boxes.append(current_box)
            
    return {"detections": detections}
            
    return {"detections": detections}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python object_detector.py <image_path>")
    else:
        res = detect_objects(sys.argv[1])
        print(json.dumps(res, indent=2))
