import torch
from transformers import pipeline
from PIL import Image
import json

def debug_scores():
    detector = pipeline(
        "zero-shot-object-detection", 
        model="google/owlvit-base-patch32",
        device=0 if torch.cuda.is_available() else -1
    )
    
    image = Image.open("nahida background.jpg").convert("RGB")
    queries = ["girl", "dress", "tree", "water", "river", "forest", "anime character"]
    
    results = detector(image, candidate_labels=queries)
    
    print("All Detection Scores:")
    for r in sorted(results, key=lambda x: x['score'], reverse=True)[:20]:
        print(f"Label: {r['label']}, Score: {r['score']:.4f}, Box: {r['box']}")

if __name__ == "__main__":
    debug_scores()
