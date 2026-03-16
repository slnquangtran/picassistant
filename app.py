import gradio as gr
import json
import asyncio
import concurrent.futures
from PIL import Image
import base64
import io
import shutil
import os

from object_detector import detect_objects
from captioning_agent import ImageCaptioningAgent
from segmentation_agent import SegmentationAgent
from depth_estimation_agent import DepthEstimationAgent
from inpainting_specialist import InpaintingSpecialist
from utils import load_image, pil_to_base64, draw_bounding_boxes

# Global Agents
captioning_agent = None
segmentation_agent = None
depth_agent = None
inpaint_agent = None

def get_agents(progress=None):
    global captioning_agent, segmentation_agent, depth_agent, inpaint_agent
    
    total_steps = 4
    current_step = 0
    
    if captioning_agent is None:
        if progress: progress(current_step/total_steps, desc="Loading Image Captioning Agent...")
        captioning_agent = ImageCaptioningAgent()
    current_step += 1
    
    if segmentation_agent is None:
        if progress: progress(current_step/total_steps, desc="Loading Segmentation Agent...")
        segmentation_agent = SegmentationAgent()
    current_step += 1
        
    if depth_agent is None:
        if progress: progress(current_step/total_steps, desc="Loading Depth Estimation Agent...")
        depth_agent = DepthEstimationAgent()
    current_step += 1
        
    if inpaint_agent is None:
        if progress: progress(current_step/total_steps, desc="Loading Inpainting Specialist (HEAVY)...")
        inpaint_agent = InpaintingSpecialist()
    current_step += 1
    
    if progress: progress(1.0, desc="All agents ready!")
    return captioning_agent, segmentation_agent, depth_agent, inpaint_agent

def clear_cache():
    global captioning_agent, segmentation_agent, depth_agent, inpaint_agent
    try:
        # Reset agents in memory
        captioning_agent = None
        segmentation_agent = None
        depth_agent = None
        inpaint_agent = None
        
        # Explicitly clear memory
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Delete HF cache
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            
        return "✅ Cache cleared and models reset. They will be re-downloaded on next use."
    except Exception as e:
        return f"❌ Error clearing cache: {str(e)}"

# Custom CSS for Premium Look
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
}

body {
    background-color: #0f172a;
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

.gradio-container {
    background: radial-gradient(circle at top right, #1e293b, #0f172a) !important;
}

.glass-panel {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 20px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-panel:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

.primary-btn {
    background: var(--primary-gradient) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    border-radius: 12px !important;
}

.primary-btn:hover {
    filter: brightness(1.2);
    transform: scale(1.02);
}

.secondary-output {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--glass-border);
}

#title-container {
    text-align: center;
    margin-bottom: 2rem;
}

h1 {
    font-weight: 600;
    letter-spacing: -0.025em;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
"""

def process_image(image, progress=gr.Progress(track_tqdm=True)):
    if image is None:
        return None, None, "", gr.update(choices=[]), None, None
    
    # Ensure agents are loaded
    progress(0, desc="Initializing AI ecosystem...")
    caption_agent, seg_agent, d_agent, _ = get_agents(progress=progress)
    
    progress(0.8, desc="Processing Image Content...")
    
    # Convert PIL to base64 for agents (some might expect base64/data URI)
    image_base64 = pil_to_base64(image)
    image_data_uri = f"data:image/png;base64,{image_base64}"
    
    try:
        # 1. Object Detection and Captioning in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_det = executor.submit(detect_objects, image_data_uri)
            future_cap = executor.submit(caption_agent.get_caption, image_data_uri)
            
            detection_result = future_det.result()
            caption_result = future_cap.result()
            
        detections = detection_result.get("detections", [])
        caption = caption_result.get("caption", "")
        
        # 2. Draw Bounding Boxes
        annotated_image = draw_bounding_boxes(image, detections)
        
        # 3. Segmentation (uses detections)
        formatted_bboxes = []
        labels = []
        for d in detections:
            xmin, ymin, w, h = d["bbox"]
            formatted_bboxes.append({
                "label": d["label"],
                "box": [xmin, ymin, xmin + w, ymin + h],
                "score": d["confidence"]
            })
            labels.append(f"{d['label']} ({d['confidence']})")
            
        segmentation_json = seg_agent.run(image_data_uri, formatted_bboxes)
        masks = json.loads(segmentation_json).get("masks", [])
        
        # 4. Depth Estimation
        depth_json = d_agent.run(image_data_uri)
        depth_map_base64 = json.loads(depth_json).get("depth_map", "")
        depth_map = load_image(depth_map_base64)
        
        return annotated_image, depth_map, caption, gr.update(choices=labels, value=labels[0] if labels else None), image, masks, labels

    except Exception as e:
        raise gr.Error(f"Error processing image: {str(e)}")

def inpaint_selected(original_image, masks, labels, selected_label, prompt, progress=gr.Progress(track_tqdm=True)):
    if not original_image or not masks or not selected_label or not prompt:
        return None
    
    # Ensure agents are loaded
    progress(0, desc="Warming up Inpainting specialist...")
    _, _, _, inpaint_sp = get_agents(progress=progress)
    
    progress(0.5, desc="Performing Stable Diffusion Inpainting...")
    
    try:
        # Find index of selected label
        idx = labels.index(selected_label)
        selected_mask_base64 = masks[idx]
        
        image_base64 = pil_to_base64(original_image)
        image_data_uri = f"data:image/png;base64,{image_base64}"
        mask_data_uri = f"data:image/png;base64,{selected_mask_base64}"
        
        result_json = inpaint_sp.run(image_data_uri, mask_data_uri, prompt)
        result_data = json.loads(result_json)
        inpainted_base64 = result_data.get("inpainted_image", "")
        
        return load_image(inpainted_base64)
    except Exception as e:
        raise gr.Error(f"Error inpainting: {str(e)}")

with gr.Blocks() as demo:
    with gr.Column(elem_id="title-container"):
        gr.Markdown("# 🌌 Interactive Image Explorer")
        gr.Markdown("Transform and explore your images with AI specialists.")

    # State
    original_image_state = gr.State()
    masks_state = gr.State()
    labels_state = gr.State()

    with gr.Row():
        with gr.Column(scale=2):
            img_input = gr.Image(label="Upload Image", type="pil", elem_classes="glass-panel")
            process_btn = gr.Button("🚀 Process Image", variant="primary", elem_classes="primary-btn")
        
        with gr.Column(scale=3):
            with gr.Row():
                annotated_output = gr.Image(label="Detected Objects", elem_classes="glass-panel secondary-output")
                depth_output = gr.Image(label="Depth Map", elem_classes="glass-panel secondary-output")
            caption_output = gr.Textbox(label="AI Perspective / Caption", placeholder="Caption will appear here...", elem_classes="glass-panel")

    with gr.Row(elem_classes="glass-panel"):
        with gr.Column():
            gr.Markdown("### ✨ Object Manipulation")
            with gr.Row():
                object_dropdown = gr.Dropdown(label="Target Object", choices=[], interactive=True)
                prompt_input = gr.Textbox(label="Inpainting Prompt", placeholder="What should replace this object?", interactive=True)
            inpaint_btn = gr.Button("🪄 Inpaint / Replace", variant="primary", elem_classes="primary-btn")
        
        with gr.Column():
            inpaint_output = gr.Image(label="Inpainted Result", elem_classes="secondary-output")

    with gr.Row(elem_classes="glass-panel"):
        with gr.Column():
            gr.Markdown("### ⚙️ System Settings")
            clear_btn = gr.Button("🧹 Uninstall / Clear Cache", variant="secondary")
            clear_output = gr.Markdown("")

    # Interactivity
    process_btn.click(
        process_image,
        inputs=[img_input],
        outputs=[annotated_output, depth_output, caption_output, object_dropdown, original_image_state, masks_state, labels_state]
    )

    inpaint_btn.click(
        inpaint_selected,
        inputs=[original_image_state, masks_state, labels_state, object_dropdown, prompt_input],
        outputs=[inpaint_output]
    )

    clear_btn.click(
        clear_cache,
        outputs=[clear_output]
    )

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Default())
