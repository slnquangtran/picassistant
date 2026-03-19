import torch
from transformers import pipeline
from PIL import Image
import numpy as np

def test_minimal():
    try:
        print("Loading pipeline...")
        detector = pipeline(
            "zero-shot-object-detection", 
            model="google/owlvit-base-patch32",
            device=-1 # Use CPU for minimal test to avoid GPU issues
        )
        
        # Create a tiny 100x100 white image
        print("Creating dummy image...")
        dummy_img = Image.fromarray(np.uint8(np.ones((100, 100, 3)) * 255))
        
        print("Running detection on dummy image...")
        results = detector(dummy_img, candidate_labels=["white square", "object"])
        print(f"Results: {results}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal()
