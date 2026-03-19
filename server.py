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
from utils import pil_to_base64, auto_generate_queries

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("capcut-vision")

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
logger.info("Initializing AI Agents...")
try:
    captioning_agent = ImageCaptioningAgent()
    segmentation_agent = SegmentationAgent()
    depth_agent = DepthEstimationAgent()
    inpaint_agent = InpaintingSpecialist()
    logger.info("All agents initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize agents: {e}")
    # We'll still start the app, but endpoints might fail

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/process-image")
async def process_image_endpoint(
    file: UploadFile = File(...),
    queries: str = Form(None),
    threshold: float = Form(0.1)
):
    try:
        logger.info(f"Processing image: {file.filename}")
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode("utf-8")
        image_data_uri = f"data:image/jpeg;base64,{image_base64}"
        
        # 1. Object Detection and Captioning in parallel
        loop = asyncio.get_event_loop()
        
        caption_task = loop.run_in_executor(None, captioning_agent.get_caption, image_data_uri)
        caption_result = await caption_task
        caption = caption_result.get("caption", "")
        
        if not queries or queries.strip() == "":
            queries = auto_generate_queries(caption)
            
        detection_task = loop.run_in_executor(None, detect_objects, image_data_uri, queries=queries, threshold=threshold)
        detection_result = await detection_task
        
        # 2. Segmentation (uses detections)
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
            "queries": queries,
            "detections": detection_result.get("detections", []),
            "caption": caption,
            "masks": segmentation_result.get("masks", []),
            "depth_map": depth_result.get("depth_map", "")
        }
    except Exception as e:
        logger.error(f"Error in /process-image: {e}")
        return {"error": str(e)}, 500

@app.post("/segment-point")
async def segment_point_endpoint(
    image: str = Form(...),
    x: float = Form(...),
    y: float = Form(...)
):
    try:
        logger.info(f"Segmenting point: {x}, {y}")
        # Call Segmentation Agent with point
        points = [[x, y]]
        result_json = segmentation_agent.run(image, points=points)
        result = json.loads(result_json)
        
        return {
            "masks": result.get("masks", []),
            "label": f"Selected Point ({x}, {y})"
        }
    except Exception as e:
        logger.error(f"Error in /segment-point: {e}")
        return {"error": str(e)}, 500

@app.post("/inpaint")
async def inpaint_endpoint(
    image: str = Form(...),
    mask: str = Form(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(None),
    guidance_scale: float = Form(7.5),
    strength: float = Form(1.0)
):
    try:
        logger.info(f"Inpainting with prompt: {prompt}, strength: {strength}")
        # Call Inpainting Agent
        result_json = inpaint_agent.run(
            image, mask, prompt, 
            negative_prompt=negative_prompt, 
            guidance_scale=guidance_scale, 
            strength=strength
        )
        result = json.loads(result_json)
        
        return {
            "inpainted_image": result.get("inpainted_image", "")
        }
    except Exception as e:
        logger.error(f"Error in /inpaint: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
