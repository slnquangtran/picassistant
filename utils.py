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
    Draws bounding boxes and labels on a PIL image with a premium look.
    """
    from PIL import ImageDraw, ImageFont, ImageColor
    import random
    
    annotated = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(annotated)
    
    # Predefined harmonious colors
    COLORS = [
        "#FF385C", "#FF9900", "#00F2FE", "#8E2DE2", "#45E3FF", 
        "#FFD700", "#39FF14", "#FF00FF", "#00FFEF", "#710193"
    ]
    
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()
        
    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        label = det["label"]
        confidence = det.get("confidence", 0)
        
        # Pick a color based on label or index
        color_hex = COLORS[hash(label) % len(COLORS)]
        rgb = ImageColor.getrgb(color_hex)
        
        # Draw soft glowing box (simulated by multiple outlines)
        for offset in range(3):
            alpha = 150 - (offset * 40)
            draw.rectangle([x-offset, y-offset, x+w+offset, y+h+offset], outline=(*rgb, alpha), width=1)
        
        draw.rectangle([x, y, x + w, y + h], outline=(*rgb, 255), width=3)
        
        # Draw label with semi-transparent background
        text = f"{label.upper()} {int(confidence*100)}%" if confidence > 0 else label.upper()
        
        try:
            t_bbox = draw.textbbox((x, y), text, font=font)
            # Add padding to text background
            t_bg = [t_bbox[0]-4, t_bbox[1]-4, t_bbox[2]+4, t_bbox[3]+4]
            draw.rectangle(t_bg, fill=(*rgb, 200))
        except AttributeError:
            tw, th = draw.textsize(text, font=font)
            draw.rectangle([x, y, x + tw, y + th], fill=(*rgb, 200))
            
        draw.text((x, y), text, fill="white", font=font)
        
    return annotated.convert("RGB")

def draw_mask_overlay(image, masks, labels=None, alpha=0.5):
    """
    Overlays multiple masks onto the image with different colors.
    """
    from PIL import ImageDraw, ImageColor
    import numpy as np
    
    annotated = image.copy().convert("RGBA")
    width, height = image.size
    
    COLORS = [
        "#FF385C", "#FF9900", "#00F2FE", "#8E2DE2", "#45E3FF", 
        "#FFD700", "#39FF14", "#FF00FF", "#00FFEF", "#710193"
    ]
    
    for i, mask_b64 in enumerate(masks):
        mask = load_image(mask_b64).convert("L")
        mask_np = np.array(mask)
        
        color_hex = COLORS[i % len(COLORS)]
        rgb = ImageColor.getrgb(color_hex)
        
        # Create a colored layer
        overlay = Image.new("RGBA", (width, height), (*rgb, int(255 * alpha)))
        
        # Apply mask to the overlay
        mask_img = Image.fromarray(mask_np)
        annotated = Image.composite(overlay, annotated, mask_img)
        
    return annotated.convert("RGB")

def auto_generate_queries(caption):
    """
    Generates a list of candidate queries for OWL-ViT based on a caption.
    """
    if not caption:
        return "person, girl, anime character, dress, hair, face, trees, forest, sky"
    
    # Common labels to always include
    base_labels = ["person", "girl", "anime character", "face", "hair"]
    
    # Process caption to find potential objects
    # Remove common filler words
    stop_words = ["a", "an", "the", "is", "are", "standing", "sitting", "lying", "in", "on", "at", "with", "around"]
    words = caption.lower().split()
    keywords = [w.strip(",.") for w in words if w.strip(",.") not in stop_words]
    
    # Combine and de-duplicate
    all_queries = base_labels + keywords
    # Keep order but remove dupes
    seen = set()
    unique_queries = [x for x in all_queries if not (x in seen or seen.add(x))]
    
    return ", ".join(unique_queries[:15]) # Limit to 15 for stability
