import io
import json
import base64
from PIL import Image
from transformers import pipeline

def load_image(image_input):
    """
    Loads an image from a file path or a base64 string.
    """
    if isinstance(image_input, str):
        if image_input.startswith("data:image") or ";base64," in image_input:
            # Handle data URI
            header, encoded = image_input.split(",", 1)
            image_data = base64.b64decode(encoded)
            return Image.open(io.BytesIO(image_data))
        try:
            # Check if it's a valid base64 string without header
            image_data = base64.b64decode(image_input)
            return Image.open(io.BytesIO(image_data))
        except Exception:
            # Treat as file path
            return Image.open(image_input)
    else:
        raise ValueError("Input must be a file path or base64 string.")

def detect_objects(image_input):
    """
    Performs object detection using DETR model and returns results in JSON format.
    """
    # 1. Load the image
    image = load_image(image_input)
    
    # 2. Initialize the object detection pipeline
    # Using facebook/detr-resnet-50 as requested
    detector = pipeline("object-detection", model="facebook/detr-resnet-50")
    
    # 3. Run inference
    results = detector(image)
    
    # 4. Filter and format results
    detections = []
    for result in results:
        if result['score'] >= 0.5:
            # DETR returns [xmin, ymin, xmax, ymax]
            # Convert to [x_min, y_min, width, height]
            box = result['box']
            xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            
            detections.append({
                "label": result['label'],
                "confidence": round(result['score'], 4),
                "bbox": [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
            })
            
    return {"detections": detections}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python object_detector.py <image_path_or_base64>")
    else:
        try:
            result = detect_objects(sys.argv[1])
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
