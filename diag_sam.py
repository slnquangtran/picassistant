import torch
from transformers import pipeline
from PIL import Image
import numpy as np

def diag():
    print("Loading SAM...")
    # use a tiny image for speed
    img = Image.new('RGB', (100, 100), color = 'red')
    segmenter = pipeline("mask-generation", model="facebook/sam-vit-base", device=0 if torch.cuda.is_available() else -1)
    
    print("Running SAM with boxes...")
    # [xmin, ymin, xmax, ymax]
    boxes = [[10, 10, 50, 50]]
    outputs = segmenter(img, input_boxes=[boxes])
    
    print("Output Type:", type(outputs))
    print("Output keys if dict:", outputs.keys() if isinstance(outputs, dict) else "Not a dict")
    if isinstance(outputs, list):
        print("Output length:", len(outputs))
        print("First element keys:", outputs[0].keys() if len(outputs)>0 else "Empty list")

if __name__ == "__main__":
    diag()
