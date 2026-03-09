import json
from transformers import pipeline
from utils import load_image, pil_to_base64, calculate_iou

class SegmentationAgent:
    def __init__(self, model_name="facebook/detr-resnet-50-panoptic"):
        """
        Initializes the segmentation pipeline.
        """
        self.segmenter = pipeline("image-segmentation", model=model_name)

    def run(self, image_input, bboxes=None):
        """
        Runs instance segmentation on the input image.
        If bboxes are provided, filters masks based on IoU.
        """
        image = load_image(image_input)
        results = self.segmenter(image)
        
        filtered_results = []
        
        if bboxes:
            # bboxes format: [{"label": "cat", "box": [xmin, ymin, xmax, ymax], "score": 0.9}, ...]
            for detection in bboxes:
                 det_box = detection['box']
                 best_mask = None
                 max_iou = 0
                 
                 for result in results:
                     # Calculate IoU between mask bounding box and detection box
                     # result['mask'] is a PIL image (binary)
                     mask = result['mask']
                     mask_bbox = mask.getbbox() # [xmin, ymin, xmax, ymax]
                     
                     if mask_bbox:
                         iou = calculate_iou(det_box, mask_bbox)
                         if iou > max_iou:
                             max_iou = iou
                             best_mask = mask
                 
                 if best_mask and max_iou > 0.5:
                     filtered_results.append(best_mask)
                 else:
                     # If no significant overlap, we might still want to add an empty mask 
                     # or handle it as per the "same order" requirement.
                     # For now, let's just add the best available or None if really bad.
                     # The prompt says: "Ensure masks are in the same order as detections if bboxes were provided."
                     filtered_results.append(best_mask) 
        else:
            filtered_results = [r['mask'] for r in results]

        # Convert masks to base64
        base64_masks = [pil_to_base64(mask) if mask else "" for mask in filtered_results]

        return json.dumps({"masks": base64_masks})

if __name__ == "__main__":
    # Example usage (commented out)
    # agent = SegmentationAgent()
    # print(agent.run("path/to/image.jpg"))
    pass
