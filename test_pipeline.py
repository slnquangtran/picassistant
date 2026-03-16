import asyncio
from app import get_agents, process_image, inpaint_selected
from PIL import Image

def test(image_path="test_image.png"):
    print("Loading agents...")
    caption_agent, seg_agent, depth_agent, inpaint_agent = get_agents(progress=lambda *args, **kwargs: None)

    # Let's see what images we have
    import os
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found.")
        return

    print(f"Loading image {image_path}...")
    img = Image.open(image_path).convert("RGB")

    print("Processing image (Detection, Captioning, Segmentation, Depth)...")
    annotated_output, depth_output, caption_output, labels_update, image_state, masks_state, labels_state = process_image(img, progress=lambda *args, **kwargs: None)

    print("Caption:", caption_output)
    print("Labels detected:", labels_state)

    target_label = None
    for l in labels_state:
        if "person" in l.lower() or "woman" in l.lower():
            target_label = l
            break

    if not target_label and labels_state:
        target_label = labels_state[0]

    if target_label:
        print(f"Inpainting target: {target_label}...")
        prompt = "a woman in a red dress in a forest"
        
        result_img = inpaint_selected(
            original_image=image_state,
            masks=masks_state,
            labels=labels_state,
            selected_label=target_label,
            prompt=prompt,
            progress=lambda *args, **kwargs: None
        )
        if result_img:
            result_img.save("output_red_dress.png")
            print("Successfully saved output_red_dress.png")
        else:
            print("Inpainting failed.")
    else:
        print("No target found for inpainting.")

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.png"
    test(img_path)
