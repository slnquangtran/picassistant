import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Upload, Image as ImageIcon, Sparkles, Layers, Box, Info } from 'lucide-react';

const API_BASE = "http://localhost:8000";

function App() {
  const [image, setImage] = useState(null);
  const [fileName, setFileName] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [selectedObjectIndex, setSelectedObjectIndex] = useState(null);
  const [inpaintPrompt, setInpaintPrompt] = useState("");
  const [inpainting, setInpainting] = useState(false);
  const [inpaintedImage, setInpaintedImage] = useState(null);
  const [showStage, setShowStage] = useState('original'); // 'original', 'depth', 'mask'
  const [imgNaturalSize, setImgNaturalSize] = useState({ width: 1, height: 1 });

  const fileInputRef = useRef(null);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setFileName(file.name);
    setLoading(true);
    setResults(null);
    setSelectedObjectIndex(null);
    setInpaintedImage(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(`${API_BASE}/process-image`, formData);
      setResults(response.data);
      setImage(response.data.image);
    } catch (error) {
      console.error("Error processing image:", error);
      alert("Failed to process image. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleInpaint = async () => {
    if (selectedObjectIndex === null || !inpaintPrompt) return;

    setInpainting(true);
    const mask = results.masks[selectedObjectIndex];

    try {
      const formData = new FormData();
      formData.append("image", results.image);
      formData.append("mask", mask);
      formData.append("prompt", inpaintPrompt);

      const response = await axios.post(`${API_BASE}/inpaint`, formData);
      setInpaintedImage(response.data.inpainted_image);
      setImage(`data:image/png;base64,${response.data.inpainted_image}`);
      setShowStage('original');
    } catch (error) {
      console.error("Error inpainting:", error);
      alert("Inpainting failed.");
    } finally {
      setInpainting(false);
    }
  };

  const renderActiveImage = () => {
    if (showStage === 'depth' && results?.depth_map) {
      return `data:image/png;base64,${results.depth_map}`;
    }
    if (showStage === 'mask' && selectedObjectIndex !== null && results?.masks[selectedObjectIndex]) {
       return `data:image/png;base64,${results.masks[selectedObjectIndex]}`;
    }
    return image;
  };

  return (
    <div className="app-container">
      <header>
        <h1>CapCut Vision</h1>
        <p className="subtitle">Interactive Multi-Agent Image Exploration & Inpainting</p>
      </header>

      {!image && !loading && (
        <div className="glass-card upload-section" onClick={() => fileInputRef.current.click()}>
          <Upload size={48} color="#9d50bb" />
          <div style={{ textAlign: 'center' }}>
            <h2>Upload an Image</h2>
            <p>Drag and drop or click to explore AI vision</p>
          </div>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            hidden 
            accept="image/*"
          />
        </div>
      )}

      {loading && (
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '4rem' }}>
          <div className="spinner"></div>
          <h2>Analyzing Image...</h2>
          <p style={{ color: 'var(--text-dim)', marginTop: '1rem' }}>Orchestrating AI agents (Detection, Captioning, Segmentation, Depth)</p>
        </div>
      )}

      {image && !loading && (
        <div className="main-grid">
          <div className="glass-card" style={{ padding: '1rem' }}>
            <div className="image-viewer">
              {inpainting && (
                <div className="loading-overlay">
                  <div className="spinner"></div>
                  <p>Inpainting with Stable Diffusion...</p>
                </div>
              )}
              <div className="image-wrapper">
                <img 
                  src={renderActiveImage()} 
                  alt="Preview" 
                  onLoad={(e) => {
                    if (showStage === 'original') {
                      setImgNaturalSize({
                        width: e.target.naturalWidth,
                        height: e.target.naturalHeight
                      });
                    }
                  }}
                />
                
                {showStage === 'original' && results?.detections && !inpaintedImage && (
                  <div className="canvas-overlay">
                    {results.detections.map((det, index) => {
                      const [x, y, w, h] = det.bbox;
                      const left = (x / imgNaturalSize.width) * 100;
                      const top = (y / imgNaturalSize.height) * 100;
                      const width = (w / imgNaturalSize.width) * 100;
                      const height = (h / imgNaturalSize.height) * 100;

                      return (
                        <div 
                          key={index}
                          className={`bounding-box ${selectedObjectIndex === index ? 'selected' : ''}`}
                          style={{
                            left: `${left}%`,
                            top: `${top}%`,
                            width: `${width}%`,
                            height: `${height}%`,
                          }}
                          onClick={() => {
                            setSelectedObjectIndex(index);
                            setShowStage('mask');
                          }}
                        >
                          <div className="bounding-box-label">
                            {det.label} {Math.round(det.confidence * 100)}%
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem', justifyContent: 'center' }}>
              <button 
                className={`btn ${showStage === 'original' ? 'btn-primary' : 'btn-secondary'}`} 
                onClick={() => setShowStage('original')}
                style={{ background: showStage === 'original' ? '' : 'rgba(255,255,255,0.1)' }}
              >
                <ImageIcon size={18} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                Original
              </button>
              <button 
                className={`btn ${showStage === 'depth' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setShowStage('depth')}
                style={{ background: showStage === 'depth' ? '' : 'rgba(255,255,255,0.1)' }}
                disabled={!results?.depth_map}
              >
                <Layers size={18} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                Depth Map
              </button>
              {selectedObjectIndex !== null && (
                <button 
                  className={`btn ${showStage === 'mask' ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setShowStage('mask')}
                  style={{ background: showStage === 'mask' ? '' : 'rgba(255,255,255,0.1)' }}
                >
                  <Sparkles size={18} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                  Active Mask
                </button>
              )}
            </div>

            {results?.caption && (
              <div className="caption-box" style={{ marginTop: '1.5rem' }}>
                <Info size={16} style={{ marginRight: '8px', marginBottom: '-2px' }} />
                {results.caption}
              </div>
            )}
          </div>

          <div className="controls-panel">
            <div className="glass-card">
              <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center' }}>
                <Box size={20} style={{ marginRight: '8px' }} />
                Detected Objects
              </h3>
              <div className="detection-list">
                {results?.detections?.map((det, index) => (
                  <div 
                    key={index} 
                    className={`detection-item ${selectedObjectIndex === index ? 'selected' : ''}`}
                    onClick={() => {
                      setSelectedObjectIndex(index);
                      setShowStage('mask');
                    }}
                  >
                    <span>{det.label}</span>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>{Math.round(det.confidence * 100)}%</span>
                  </div>
                ))}
                {(!results?.detections || results.detections.length === 0) && (
                  <p style={{ color: 'var(--text-dim)', textAlign: 'center' }}>No objects detected.</p>
                )}
              </div>

              {selectedObjectIndex !== null && !inpaintedImage && (
                <div className="inpaint-form">
                  <h4 style={{ marginTop: '1rem' }}>Inpaint Selected Object</h4>
                  <input 
                    type="text" 
                    placeholder="Wanna change it to something else? (e.g. 'a robotic cat')"
                    value={inpaintPrompt}
                    onChange={(e) => setInpaintPrompt(e.target.value)}
                  />
                  <button className="btn btn-primary" onClick={handleInpaint} disabled={!inpaintPrompt || inpainting}>
                    <Sparkles size={18} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                    {inpainting ? 'Inpainting...' : 'Run Inpainting'}
                  </button>
                </div>
              )}

              {inpaintedImage && (
                <div style={{ marginTop: '1.5rem', textAlign: 'center' }}>
                  <p style={{ color: 'var(--accent)', fontWeight: '600' }}>✨ Final Result Ready!</p>
                  <button className="btn btn-primary" style={{ marginTop: '1rem', width: '100%' }} onClick={() => window.location.reload()}>
                    Reset Experience
                  </button>
                </div>
              )}
            </div>
            
            <div className="glass-card" style={{ marginTop: '1rem', fontSize: '0.9rem', color: 'var(--text-dim)' }}>
               <h4 style={{ color: 'white', marginBottom: '0.5rem' }}>System Status</h4>
               <ul style={{ listSetStyle: 'none', padding: 0 }}>
                 <li style={{ color: 'var(--accent)' }}>✓ Detection Agent: Active</li>
                 <li style={{ color: 'var(--accent)' }}>✓ Captioning Agent: Active</li>
                 <li style={{ color: 'var(--accent)' }}>✓ Segmentation Agent: Active</li>
                 <li style={{ color: 'var(--accent)' }}>✓ Depth Agent: Active</li>
                 <li style={{ color: results ? 'var(--accent)' : 'var(--text-dim)' }}>
                   {results ? '✓ Inpainting Agent: Ready' : 'Inpainting Agent: Loading...'}
                 </li>
               </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
