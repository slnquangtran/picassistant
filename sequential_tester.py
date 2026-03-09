import torch
import gc
import json
import os
from PIL import Image, ImageDraw
import io
import base64

# Dummy image for testing
def create_dummy_image(path="test_image.png"):
    img = Image.new('RGB', (512, 512), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,10), "Test Image", fill=(255,255,0))
    img.save(path)
    return path

def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def print_memory_usage():
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Current Memory Usage: {mem_info.rss / 1024 / 1024:.2f} MB")

def clear_memory(model=None):
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleared.")
    print_memory_usage()

def test_detection(img_path):
    print("\n--- Testing Object Detection ---")
    from object_detector import detect_objects
    result = detect_objects(img_path)
    print(f"Detected: {[d['label'] for d in result['detections']]}")
    # Note: detect_objects doesn't return the model, it creates it inside.
    # We might need to rely on gc.collect() or modify the script if it persists.
    clear_memory()

def test_segmentation(img_path):
    print("\n--- Testing Segmentation ---")
    from segmentation_agent import SegmentationAgent
    agent = SegmentationAgent()
    result_json = agent.run(img_path)
    result = json.loads(result_json)
    print(f"Masks generated: {len(result['masks'])}")
    clear_memory(agent)

def test_depth(img_path):
    print("\n--- Testing Depth Estimation ---")
    from depth_estimation_agent import DepthEstimationAgent
    agent = DepthEstimationAgent()
    result_json = agent.run(img_path)
    result = json.loads(result_json)
    print(f"Depth map generated: {'depth_map' in result}")
    clear_memory(agent)

def test_caption(img_path):
    print("\n--- Testing Captioning ---")
    from captioning_agent import ImageCaptioningAgent
    agent = ImageCaptioningAgent()
    result = agent.get_caption(img_path)
    print(f"Caption: {result.get('caption', 'Error')}")
    clear_memory(agent)

def test_inpainting(img_path):
    print("\n--- Testing Inpainting ---")
    from inpainting_specialist import InpaintingSpecialist
    # Create a dummy mask
    mask = Image.new('L', (512, 512), color=0)
    d = ImageDraw.Draw(mask)
    d.rectangle([100, 100, 200, 200], fill=255)
    mask_path = "test_mask.png"
    mask.save(mask_path)
    
    agent = InpaintingSpecialist()
    try:
        result_json = agent.run(img_path, mask_path, "a red ball")
        result = json.loads(result_json)
        print(f"Inpainted image generated: {'inpainted_image' in result}")
    except Exception as e:
        print(f"Inpainting failed: {e}")
    finally:
        clear_memory(agent)
        if os.path.exists(mask_path):
            os.remove(mask_path)

if __name__ == "__main__":
    img_path = create_dummy_image()
    
    try:
        # test_detection(img_path)
        # test_depth(img_path)
        # test_caption(img_path)
        # test_segmentation(img_path)
        test_inpainting(img_path) # We know this is very heavy, test last
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)
