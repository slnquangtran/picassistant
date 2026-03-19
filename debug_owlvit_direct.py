import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import json

def debug_direct():
    print("Loading Processor and Model (google/owlvit-base-patch32)...")
    try:
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        
        image = Image.open("nahida background.jpg").convert("RGB")
        # Structure as list of lists for prompt batching
        texts = [["a girl", "white hair", "green dress", "tree", "forest", "river", "water"]]
        
        print("Running inference...")
        inputs = processor(text=texts, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO format
        results = processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.01)
        
        print(f"Post-processed results: {len(results)} items")
        
        i = 0  # Only one image
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        
        if len(boxes) == 0:
            print("No boxes found even at 0.1 threshold.")
        
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_direct()
