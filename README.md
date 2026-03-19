# 🌌 CapCut Vision - AI Image Exploration Ecosystem

CapCut Vision is a premium, multi-agent AI system designed for advanced image exploration, automated captioning, depth estimation, and intelligent inpainting. It leverages a suite of specialized AI agents to provide a seamless and interactive "computer vision" experience.

![Preview](nahida_red_dress.png) *(Example output)*

## ✨ Key Features

- **Object Detection & Recognition**: Automatically identifies multiple objects in an image with high precision using DETR.
- **Interactive Segmentation**: Uses Segment Anything Model (SAM) to generate precise masks for any detected object.
- **Depth Estimation**: Visualizes the scene's spatial structure with DPT-Large.
- **AI-Driven Captioning**: Generates descriptive "AI perspectives" for your images using BLIP.
- **Intelligent Inpainting**: Transform any selected object into something entirely new using Stable Diffusion Inpainting.
- **Premium React UI**: A sleek, dark-themed, glassmorphic interface for a high-end user experience.

## 🛠️ Technology Stack

- **Backend**: FastAPI, PyTorch, Transformers (Hugging Face), Diffusers, Uvicorn.
- **Frontend**: React (Vite), Axios, Lucide Icons, Vanilla CSS (Premium Tokens).
- **AI Models**:
  - `facebook/detr-resnet-50` (Detection)
  - `facebook/sam-vit-base` (Segmentation)
  - `Intel/dpt-large` (Depth)
  - `Salesforce/blip-image-captioning-base` (Captioning)
  - `runwayml/stable-diffusion-inpainting` (Inpainting)

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js & npm
- CUDA-compatible GPU (Highly recommended for fluid performance)

### Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Frontend dependencies**:
   ```bash
   npm install
   ```

### Running the Project

The easiest way to start both the backend and frontend is using the provided PowerShell script:

```powershell
./run_project.ps1
```

Alternatively, you can start them manually:

- **Backend**: `python server.py` (Starts on `http://localhost:8000`)
- **Frontend**: `npm run dev` (Starts on `http://localhost:5173`)

## ⚙️ Project Structure

- `server.py`: FastAPI server orchestrating the AI agents.
- `app.py`: Alternative Gradio interface for debugging/standalone use.
- `src/App.jsx`: Main React application logic.
- `src/index.css`: Premium design system and styles.
- `*_agent.py`: Specialized wrappers for individual AI models.
- `utils.py`: Shared utilities for image processing and base64 handling.

## 🧼 Management

To clear the AI model cache and free up disk space:
```bash
python verify_uninstall.py
```
*(Or use the button in the Gradio `app.py` interface)*
