import torch
from transformers import pipeline
from PIL import Image
import numpy as np

def debug_owlvit():
    try:
        print("Loading OWL-ViT v1...")
        detector = pipeline(
            "zero-shot-object-detection", 
            model="google/owlvit-base-patch32",
            device=-1
        )
        
        img = Image.open("nahida background.jpg").convert("RGB")
        # Try very broad queries
        queries = ["person", "girl", "tree", "forest", "shoes", "cloth", "face", "hair"]
        
        print(f"Running detection on {img.size} image...")
        results = detector(img, candidate_labels=queries)
        
        if not results:
            print("No results returned at all (None or empty list)")
        else:
            print(f"Found {len(results)} raw results.")
            # Sort by score and show top 10 regardless of threshold
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            for i, r in enumerate(sorted_results[:20]):
                print(f"{i+1}. Label: {r['label']}, Score: {r['score']:.4f}")
                
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_owlvit()
