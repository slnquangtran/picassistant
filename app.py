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
from utils import load_image, pil_to_base64, draw_bounding_boxes, draw_mask_overlay, auto_generate_queries

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

CSS = """
/* Futuristic AI Orchestrator Theme */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=JetBrains+Mono&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%);
    --accent: #00f2fe;
    --glass-bg: rgba(15, 23, 42, 0.7);
    --glass-border: rgba(255, 255, 255, 0.1);
    --neon-shadow: 0 0 15px rgba(142, 45, 226, 0.3);
}

body {
    background-color: #020617;
    background-image: 
        radial-gradient(at 0% 0%, rgba(142, 45, 226, 0.1) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(0, 242, 254, 0.05) 0px, transparent 50%);
    color: #f8fafc;
    font-family: 'Outfit', sans-serif;
}

.gradio-container {
    background: transparent !important;
    border: none !important;
}

.glass-panel {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 24px !important;
    padding: 24px !important;
    box-shadow: var(--neon-shadow) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.glass-panel:hover {
    border-color: rgba(255, 255, 255, 0.2) !important;
    transform: translateY(-4px);
}

.primary-btn {
    background: var(--primary-gradient) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border-radius: 14px !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 15px rgba(74, 0, 224, 0.4) !important;
}

.primary-btn:hover {
    filter: brightness(1.2);
    box-shadow: 0 6px 20px rgba(74, 0, 224, 0.6) !important;
}

h1 {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(to right, #fff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
}

.caption-box {
    background: rgba(0, 242, 254, 0.05);
    border-left: 4px solid var(--accent);
    padding: 1rem;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
}

.object-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.2s;
}

.object-card:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: var(--accent);
}

.confidence-badge {
    background: var(--primary-gradient);
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
"""

def process_image(image, queries=None, threshold=0.1, show_masks=False, progress=gr.Progress(track_tqdm=True)):
    if image is None:
        return None, None, "", gr.update(choices=[]), None, None, [], [], ""
    
    try:
        # Ensure agents are loaded
        progress(0, desc="Initializing AI ecosystem...")
        caption_agent, seg_agent, d_agent, _ = get_agents(progress=progress)
        
        progress(0.4, desc="Interpreting Scene...")
        image_base64 = pil_to_base64(image)
        image_data_uri = f"data:image/png;base64,{image_base64}"
        
        caption_result = caption_agent.get_caption(image_data_uri)
        caption = caption_result.get("caption", "")
        
        # Auto-generate queries if none provided
        if not queries or queries.strip() == "":
            queries = auto_generate_queries(caption)
        
        progress(0.6, desc="Scanning for Objects...")
        detection_result = detect_objects(image_data_uri, queries=queries, threshold=threshold)
        detections = detection_result.get("detections", [])
        
        # 3. Segmentation (uses detections)
        formatted_bboxes = []
        labels = []
        object_explorer_html = "<div class='object-explorer'>"
        
        for d in detections:
            xmin, ymin, w, h = d["bbox"]
            formatted_bboxes.append({
                "label": d["label"],
                "box": [xmin, ymin, xmin + w, ymin + h],
                "score": d["confidence"]
            })
            labels.append(f"{d['label']} ({d['confidence']})")
            
            # Simple HTML card for Explorer
            conf_pct = int(d['confidence'] * 100)
            object_explorer_html += f"""
            <div class='object-card'>
                <span><strong>{d['label']}</strong></span>
                <span class='confidence-badge'>{conf_pct}% Match</span>
            </div>
            """
        object_explorer_html += "</div>"
            
        segmentation_json = seg_agent.run(image_data_uri, formatted_bboxes)
        masks = json.loads(segmentation_json).get("masks", [])
        
        # Determine which image to show as "Annotated"
        if show_masks and masks:
            display_image = draw_mask_overlay(image, masks)
            display_image = draw_bounding_boxes(display_image, detections)
        else:
            display_image = draw_bounding_boxes(image, detections)
        
        progress(0.9, desc="Calculating Depth Latencies...")
        depth_json = d_agent.run(image_data_uri)
        depth_map_base64 = json.loads(depth_json).get("depth_map", "")
        depth_map = load_image(depth_map_base64)
        
        return display_image, depth_map, caption, gr.update(choices=labels, value=labels[0] if labels else None), image, masks, labels, queries, object_explorer_html
    except Exception as e:
        raise gr.Error(f"Error processing image: {str(e)}")
    except Exception as e:
        raise gr.Error(f"Error processing image: {str(e)}")

def handle_select(evt: gr.SelectData, original_image, masks, labels, progress=gr.Progress(track_tqdm=True)):
    """
    Handles click on the main image for interactive segmentation.
    """
    if original_image is None:
        return gr.update(), masks, labels
    
    try:
        progress(0, desc="Precise SAM Selection...")
        _, seg_agent, _, _ = get_agents(progress=progress)
        
        # evt.index gives [x, y]
        x, y = evt.index
        image_data_uri = f"data:image/png;base64,{pil_to_base64(original_image)}"
        
        # Run SAM with points
        points = [[x, y]]
        res_json = seg_agent.run(image_data_uri, points=points)
        new_masks = json.loads(res_json).get("masks", [])
        
        if new_masks:
            new_mask = new_masks[0]
            new_label = f"Selected Point ({x}, {y})"
            
            updated_masks = (masks or []) + [new_mask]
            updated_labels = (labels or []) + [new_label]
            
            return gr.update(choices=updated_labels, value=new_label), updated_masks, updated_labels
        
        return gr.update(), masks, labels
    except Exception as e:
        raise gr.Error(f"Error in precise selection: {str(e)}")

def inpaint_selected(original_image, masks, labels, selected_label, prompt, negative_prompt=None, guidance_scale=7.5, strength=1.0, progress=gr.Progress(track_tqdm=True)):
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
        
        result_json = inpaint_sp.run(image_data_uri, mask_data_uri, prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, strength=strength)
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
            with gr.Accordion("Detection Settings", open=True):
                queries_input = gr.Textbox(
                    label="OWL-ViT Queries (Comma separated)", 
                    placeholder="e.g. girl, dress, trees, hair",
                    value="person, girl, anime character, dress, hair, face, tree, forest, sky"
                )
                threshold_slider = gr.Slider(label="Confidence Threshold", minimum=0.01, maximum=1.0, value=0.1, step=0.01)
            process_btn = gr.Button("🚀 Process Image", variant="primary", elem_classes="primary-btn")
        
        with gr.Column(scale=3):
            with gr.Row():
                annotated_output = gr.Image(label="Object Explorer / Annotation", elem_classes="glass-panel secondary-output")
                depth_output = gr.Image(label="Depth Map", elem_classes="glass-panel secondary-output")
            
            with gr.Row():
                with gr.Column(scale=2):
                    caption_output = gr.Textbox(label="AI Perspective / Caption", placeholder="Caption will appear here...", elem_classes="glass-panel")
                with gr.Column(scale=1):
                    show_masks_toggle = gr.Checkbox(label="Visualize Semantic Masks", value=True)
            
            with gr.Accordion("Named Object Explorer", open=True, elem_classes="glass-panel"):
                object_explorer_view = gr.HTML(label="Detected Objects List", value="Upload an image to name objects...")

    with gr.Row(elem_classes="glass-panel"):
        with gr.Column():
            gr.Markdown("### ✨ Object Manipulation")
            with gr.Row():
                object_dropdown = gr.Dropdown(label="Target Object", choices=[], interactive=True)
                prompt_input = gr.Textbox(label="Inpainting Prompt", placeholder="What should replace this object?", interactive=True)
            
            with gr.Accordion("Advanced Inpainting Settings", open=False):
                neg_prompt_input = gr.Textbox(label="Negative Prompt", placeholder="What to avoid...", value="blurry, ugly, distorted, low quality")
                with gr.Row():
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=7.5, step=0.5)
                    strength = gr.Slider(label="Strength (Change level)", minimum=0.0, maximum=1.0, value=1.0, step=0.05)
            
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
        inputs=[img_input, queries_input, threshold_slider, show_masks_toggle],
        outputs=[annotated_output, depth_output, caption_output, object_dropdown, original_image_state, masks_state, labels_state, queries_input, object_explorer_view]
    )

    # Interactive click selection on the PREVIEW image (once processed)
    annotated_output.select(
        handle_select,
        inputs=[original_image_state, masks_state, labels_state],
        outputs=[object_dropdown, masks_state, labels_state]
    )

    inpaint_btn.click(
        inpaint_selected,
        inputs=[original_image_state, masks_state, labels_state, object_dropdown, prompt_input, neg_prompt_input, guidance_scale, strength],
        outputs=[inpaint_output]
    )

    clear_btn.click(
        clear_cache,
        outputs=[clear_output]
    )

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Default())
