from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import asyncio
import base64
import io
from PIL import Image
from object_detector import detect_objects
from captioning_agent import ImageCaptioningAgent
from segmentation_agent import SegmentationAgent
from depth_estimation_agent import DepthEstimationAgent
from inpainting_specialist import InpaintingSpecialist
from utils import pil_to_base64

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agents
captioning_agent = ImageCaptioningAgent()
segmentation_agent = SegmentationAgent()
depth_agent = DepthEstimationAgent()
# Inpainting agent is initialized on demand or globally? 
# Stable Diffusion is heavy, let's initialize it.
inpaint_agent = InpaintingSpecialist()

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    image_base64 = base64.b64encode(contents).decode("utf-8")
    image_data_uri = f"data:image/jpeg;base64,{image_base64}"
    
    # 1. Object Detection and Captioning in parallel
    loop = asyncio.get_event_loop()
    
    detection_task = loop.run_in_executor(None, detect_objects, image_data_uri)
    caption_task = loop.run_in_executor(None, captioning_agent.get_caption, image_data_uri)
    
    detection_result, caption_result = await asyncio.gather(detection_task, caption_task)
    
    # 2. Segmentation (uses detections)
    # The segmentation agent expects bboxes in a specific format if provided.
    # From segmentation_agent.py: bboxes format: [{"label": "cat", "box": [xmin, ymin, xmax, ymax], "score": 0.9}, ...]
    # But detector returns {"detections": [{"label": "cat", "confidence": 0.9, "bbox": [xmin, ymin, w, h]}, ...]}
    
    formatted_bboxes = []
    for d in detection_result.get("detections", []):
        xmin, ymin, w, h = d["bbox"]
        formatted_bboxes.append({
            "label": d["label"],
            "box": [xmin, ymin, xmin + w, ymin + h],
            "score": d["confidence"]
        })
    
    segmentation_json = segmentation_agent.run(image_data_uri, formatted_bboxes)
    segmentation_result = json.loads(segmentation_json)
    
    # 3. Depth Estimation
    depth_json = depth_agent.run(image_data_uri)
    depth_result = json.loads(depth_json)
    
    return {
        "image": image_data_uri,
        "detections": detection_result.get("detections", []),
        "caption": caption_result.get("caption", ""),
        "masks": segmentation_result.get("masks", []),
        "depth_map": depth_result.get("depth_map", "")
    }

@app.post("/inpaint")
async def inpaint(
    image: str = Form(...),
    mask: str = Form(...),
    prompt: str = Form(...)
):
    # Call Inpainting Agent
    result_json = inpaint_agent.run(image, mask, prompt)
    result = json.loads(result_json)
    
    return {
        "inpainted_image": result.get("inpainted_image", "")
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
