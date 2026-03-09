import base64
import io
from PIL import Image

def load_image(image_input):
    """
    Loads a PIL image from a file path or base64 string.
    """
    if isinstance(image_input, str):
        if image_input.startswith("data:image"):
            # Handle data URI
            header, encoded = image_input.split(",", 1)
            image_data = base64.b64decode(encoded)
            return Image.open(io.BytesIO(image_data))
        elif len(image_input) > 255 or "\n" in image_input:
            # Likely a raw base64 string
            image_data = base64.b64decode(image_input)
            return Image.open(io.BytesIO(image_data))
        else:
            # Likely a file path
            return Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        return image_input
    else:
        raise ValueError("Unsupported image input type")

def pil_to_base64(image):
    """
    Converts a PIL image to a base64-encoded PNG string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Format: [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - intersection_area
    
    if union_area == 0:
        return 0
        
    return intersection_area / union_area

def draw_bounding_boxes(image, detections):
    """
    Draws bounding boxes and labels on a PIL image.
    :param image: PIL Image object.
    :param detections: List of detections in format [{"label": "cat", "bbox": [x, y, w, h]}, ...]
    :return: Annotated PIL Image.
    """
    from PIL import ImageDraw, ImageFont
    
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        
    for det in detections:
        x, y, w, h = det["bbox"]
        label = det["label"]
        confidence = det.get("confidence", "")
        
        # Draw box
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        
        # Draw label
        text = f"{label} {confidence}" if confidence != "" else label
        
        # Use textbbox if available (higher PIL versions)
        try:
            bbox = draw.textbbox((x, y), text, font=font)
            draw.rectangle(bbox, fill="red")
        except AttributeError:
            # Fallback for older PIL
            tw, th = draw.textsize(text, font=font)
            draw.rectangle([x, y, x + tw, y + th], fill="red")
            
        draw.text((x, y), text, fill="white", font=font)
        
    return annotated
