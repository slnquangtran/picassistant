import os
import json
import base64
from PIL import Image, ImageDraw
import torch
from app import get_agents, process_image, inpaint_selected
from utils import load_image, pil_to_base64

# Configuration
IMAGE_PATH = "nahida background.jpg"
OUTPUT_DIR = "test_outputs"
PROMPT = "vibrant red dress for the character, intricate silk texture, highly detailed anime style"
NEGATIVE_PROMPT = "blue dress, green dress, character face change, person change, ugly, blurry"

def run_comprehensive_test():
    # 1. Setup Output Directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found at {IMAGE_PATH}")
        return

    # 2. Initialize Agents
    print("--- [1/6] Initializing AI Agents ---")
    caption_agent, seg_agent, depth_agent, inpaint_agent = get_agents(progress=lambda step, desc: print(f"  {desc}"))

    # 3. Load Image
    img = Image.open(IMAGE_PATH).convert("RGB")
    print(f"--- [2/6] Processing Image: {IMAGE_PATH} ---")

    # 4. Run Main Pipeline (Detection, Captioning, Segmentation, Depth)
    # Note: process_image now returns 9 values
    queries = "person, girl, anime character, dress, hair, face, tree, forest, sky"
    results = process_image(image=img, queries=queries, threshold=0.1, show_masks=True, progress=lambda step, desc: print(f"  {desc}"))
    annotated_img, depth_map, caption, _, _, masks, labels, _, _ = results

    # 5. Save Intermediate Results
    print("--- [3/6] Saving Intermediate Results ---")
    
    # Save Caption
    with open(os.path.join(OUTPUT_DIR, "caption.txt"), "w") as f:
        f.write(caption)
    print(f"  Saved caption to {OUTPUT_DIR}/caption.txt")

    # Save Annotated Image
    if annotated_img:
        annotated_img.save(os.path.join(OUTPUT_DIR, "detected_objects.png"))
        print(f"  Saved annotated image to {OUTPUT_DIR}/detected_objects.png")

    # Save Depth Map
    if depth_map:
        depth_map.save(os.path.join(OUTPUT_DIR, "depth_map.png"))
        print(f"  Saved depth map to {OUTPUT_DIR}/depth_map.png")

    # Save Masks
    if masks:
        print(f"  Found {len(masks)} masks. Saving...")
        for i, mask_b64 in enumerate(masks):
            mask_img = load_image(mask_b64)
            label_name = labels[i].split(' ')[0] if i < len(labels) else "auto_mask"
            mask_img.save(os.path.join(OUTPUT_DIR, f"mask_{i}_{label_name}.png"))
    
    if not labels:
        print("  ! No objects detected by DETR. Creating a manual mask for inpainting test.")
        # Create a simple box mask in the center for testing
        w, h = img.size
        manual_mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(manual_mask)
        # Assuming Nahida is roughly in the center
        draw.rectangle([w//4, h//4, 3*w//4, 3*h//4], fill=255)
        manual_mask.save(os.path.join(OUTPUT_DIR, "manual_test_mask.png"))
        masks = [pil_to_base64(manual_mask)]
        labels = ["manual_box"]

    # 6. Run Inpainting
    print("--- [4/6] Running Inpainting ---")
    # For testing, we'll inpaint the first mask found (or the manual one)
    try:
        inpainted_img = inpaint_selected(
            original_image=img,
            masks=masks,
            labels=labels,
            selected_label=labels[0],
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=9.0,
            strength=0.7,
            progress=lambda step, desc: print(f"  {desc}")
        )
        if inpainted_img:
            inpainted_img.save(os.path.join(OUTPUT_DIR, "inpainted_result.png"))
            print(f"--- [5/6] SUCCESS: Final result saved to {OUTPUT_DIR}/inpainted_result.png ---")
        else:
            print("--- [5/6] FAILED: Inpainting returned None ---")
    except Exception as e:
        print(f"--- [5/6] FAILED: Inpainting error: {str(e)} ---")

    print("--- [6/6] Comprehensive Test Complete ---")

if __name__ == "__main__":
    run_comprehensive_test()
