import json
from object_detector import detect_objects

IMAGE_PATH = "nahida background.jpg"

def debug_detection():
    print(f"Testing OWL-ViT detection on {IMAGE_PATH}...")
    try:
        results = detect_objects(IMAGE_PATH)
        print("Detections found:")
        print(json.dumps(results, indent=2))
        
        if not results.get("detections"):
            print("No detections found with default threshold.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_detection()
