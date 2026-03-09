import os
import json
from PIL import Image, ImageDraw
from segmentation_agent import SegmentationAgent
from depth_estimation_agent import DepthEstimationAgent
from utils import pil_to_base64

def create_dummy_image(path="test_image.png"):
    img = Image.new('RGB', (100, 100), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,10), "Hello", fill=(255,255,0))
    img.save(path)
    return path

def test_agents():
    test_image_path = create_dummy_image()
    
    print("Testing Segmentation Agent...")
    seg_agent = SegmentationAgent()
    # Test without bboxes
    seg_output = seg_agent.run(test_image_path)
    seg_data = json.loads(seg_output)
    print(f"Segmentation masks found: {len(seg_data['masks'])}")
    
    # Test with bboxes
    dummy_bboxes = [{"label": "dummy", "box": [0, 0, 50, 50], "score": 1.0}]
    seg_output_bbox = seg_agent.run(test_image_path, bboxes=dummy_bboxes)
    seg_data_bbox = json.loads(seg_output_bbox)
    print(f"Segmentation masks with bboxes: {len(seg_data_bbox['masks'])}")

    print("\nTesting Depth Estimation Agent...")
    depth_agent = DepthEstimationAgent()
    depth_output = depth_agent.run(test_image_path)
    depth_data = json.loads(depth_output)
    print(f"Depth map generated: {'depth_map' in depth_data}")

    # Cleanup
    # (Removed to avoid issues between tests)
    pass

if __name__ == "__main__" :
    test_image_path = "test_image.png"
    try:
        test_agents()
    except Exception as e:
        print(f"Error during testing: {e}")
    finally:
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
