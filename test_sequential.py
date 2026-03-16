import os
import gc
import json
import torch
from app import pil_to_base64, load_image
from PIL import Image

def test(image_path="test_image.png"):
    print(f"Testing on image: {image_path}")
    if not os.path.exists(image_path):
        print("Image not found.")
        return

    img = Image.open(image_path).convert("RGB")
    image_base64 = pil_to_base64(img)
    image_data_uri = f"data:image/png;base64,{image_base64}"

    # 1. Captioning
    print("Loading Captioning Agent...")
    from captioning_agent import ImageCaptioningAgent
    cap_agent = ImageCaptioningAgent()
    caption_res = cap_agent.get_caption(image_data_uri)
    print("Caption:", caption_res)
    del cap_agent
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 2. Object Detection
    print("Loading Object Detector...")
    from object_detector import detect_objects
    det_res = detect_objects(image_data_uri)
    detections = det_res.get("detections", [])
    print("Detections found:", len(detections))
    
    # 3. Segmentation
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
    print("Masks generated:", len(masks))
    
    del seg_agent
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Find Target Label (person/woman or dress)
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
        print("Loading Inpainting Agent...")
        from inpainting_specialist import InpaintingSpecialist
        inpaint_agent = InpaintingSpecialist()
        
        prompt = "a woman in a red dress in a forest"
        mask_data_uri = f"data:image/png;base64,{target_mask}"
        
        res_json = inpaint_agent.run(image_data_uri, mask_data_uri, prompt)
        res_data = json.loads(res_json)
        out_base64 = res_data.get("inpainted_image", "")
        
        if out_base64:
            out_img = load_image(out_base64)
            out_img.save("output_red_dress.png")
            print("Successfully saved output_red_dress.png")
        else:
            print("Inpainting failed.")
            
        del inpaint_agent
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("No target found for inpainting.")

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.png"
    test(img_path)
