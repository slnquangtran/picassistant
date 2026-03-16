import json
from app import pil_to_base64, load_image
from PIL import Image

def test(image_path="test_image.png"):
    print(f"Testing on image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    image_base64 = pil_to_base64(img)
    image_data_uri = f"data:image/png;base64,{image_base64}"

    print("Loading Object Detector...")
    from object_detector import detect_objects
    det_res = detect_objects(image_data_uri)
    detections = det_res.get("detections", [])
    
    print("Loading Segmentation Agent...")
    from segmentation_agent import SegmentationAgent
    seg_agent = SegmentationAgent()
    
    formatted_bboxes = []
    labels = []
    for d in detections:
        xmin, ymin, w, h = d["bbox"]
        formatted_bboxes.append({
            "label": d["label"],
            "box": [xmin, ymin, xmin + w, ymin + h],
            "score": d["confidence"]
        })
        labels.append(f"{d['label']} ({d['confidence']})")
        
    seg_json = seg_agent.run(image_data_uri, formatted_bboxes)
    masks = json.loads(seg_json).get("masks", [])
    
    target_label = None
    target_mask = None
    for i, l in enumerate(labels):
        if "person" in l.lower() or "woman" in l.lower() or "dress" in l.lower():
            target_label = l
            target_mask = masks[i]
            break

    if not target_label and labels:
        target_label = labels[0]
        target_mask = masks[0]

    if target_label:
        print(f"Inpainting target: {target_label}...")
        mask_img = load_image(target_mask)
        mask_img.save("target_mask.png")
        print("Saved target_mask.png")
        
        # Check pixel values
        import numpy as np
        arr = np.array(mask_img.convert("L"))
        print(f"Mask Min: {np.min(arr)}, Max: {np.max(arr)}, Mean: {np.mean(arr):.2f}")
    else:
        print("No target found for inpainting.")

if __name__ == "__main__":
    test(r"C:\Users\Quang\.gemini\antigravity\brain\0c6d53b7-4868-477b-adbe-c23605ef61aa\woman_green_dress_1773624933060.png")
