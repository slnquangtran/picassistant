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
  const [negativePrompt, setNegativePrompt] = useState("different face, different hairstyle, ugly, blurry, different person, distorted features");
  const [strength, setStrength] = useState(0.7);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [inpainting, setInpainting] = useState(false);
  const [inpaintedImage, setInpaintedImage] = useState(null);
  const [queries, setQueries] = useState("person, girl, anime character, dress, hair, face, tree, forest, sky");
  const [threshold, setThreshold] = useState(0.1);
  const [showStage, setShowStage] = useState('original'); // 'original', 'depth', 'mask'
  const [imgNaturalSize, setImgNaturalSize] = useState({ width: 1, height: 1 });

  const [systemStatus, setSystemStatus] = useState({ backend: 'checking', agents: 'checking' });

  const fileInputRef = useRef(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await axios.get(`${API_BASE}/health`);
        if (response.data.status === 'healthy') {
          setSystemStatus({ backend: 'active', agents: 'ready' });
        } else {
          setSystemStatus({ backend: 'active', agents: 'unhealthy' });
        }
      } catch (error) {
        setSystemStatus({ backend: 'inactive', agents: 'offline' });
      }
    };
    checkHealth();
  }, []);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setFileName(file.name);
    setLoading(true);
    setResults(null);
    setSelectedObjectIndex(null);
    setInpaintedImage(null);
    setInpaintPrompt("");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("queries", queries);
    formData.append("threshold", threshold);

    try {
      const response = await axios.post(`${API_BASE}/process-image`, formData);
      if (response.data.error) {
        alert(`Backend error: ${response.data.error}`);
      } else {
        setResults(response.data);
        setImage(response.data.image);
        if (response.data.queries) {
          setQueries(response.data.queries);
        }
      }
    } catch (error) {
      console.error("Error processing image:", error);
      alert("Failed to process image. Make sure the backend is running and models are loaded.");
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
      formData.append("negative_prompt", negativePrompt);
      formData.append("strength", strength);
      formData.append("guidance_scale", guidanceScale);

      const response = await axios.post(`${API_BASE}/inpaint`, formData);
      if (response.data.error) {
        alert(`Backend error: ${response.data.error}`);
      } else {
        setInpaintedImage(response.data.inpainted_image);
        setImage(`data:image/png;base64,${response.data.inpainted_image}`);
        setShowStage('original');
      }
    } catch (error) {
      console.error("Error inpainting:", error);
      alert("Inpainting failed.");
    } finally {
      setInpainting(false);
    }
  };

  const handleImageClick = async (event) => {
    if (!results || inpainting || loading) return;

    const rect = event.target.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Convert display coordinates to natural image coordinates
    const scaleX = imgNaturalSize.width / rect.width;
    const scaleY = imgNaturalSize.height / rect.height;
    const naturalX = x * scaleX;
    const naturalY = y * scaleY;

    setInpainting(true); // Reuse inpainting state as 'processing' for now
    try {
      const formData = new FormData();
      formData.append("image", results.image);
      formData.append("x", naturalX);
      formData.append("y", naturalY);

      const response = await axios.post(`${API_BASE}/segment-point`, formData);
      if (response.data.error) {
        alert(`Segmentation error: ${response.data.error}`);
      } else {
        const newMask = response.data.masks[0];
        const newLabel = response.data.label;
        
        // Add to results
        setResults(prev => ({
          ...prev,
          masks: [...prev.masks, newMask],
          detections: [...prev.detections, {
            label: newLabel,
            confidence: 1.0,
            bbox: [naturalX - 10, naturalY - 10, 20, 20]
          }]
        }));
        
        setSelectedObjectIndex(results.detections.length);
        setShowStage('mask');
      }
    } catch (error) {
      console.error("Error segmenting point:", error);
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
          <p style={{ color: 'var(--accent)', fontSize: '0.8rem', marginTop: '0.5rem' }}>This may take 10-20 seconds on first run</p>
        </div>
      )}

      {image && !loading && (
        <div className="main-grid">
          <div className="glass-card" style={{ padding: '1.5rem' }}>
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
                   onClick={handleImageClick}
                   style={{ cursor: results ? 'crosshair' : 'default' }}
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
                <strong>AI Caption:</strong> {results.caption}
              </div>
            )}
            
            <button 
              className="btn btn-secondary" 
              style={{ marginTop: '1rem', width: '100%', background: 'rgba(255,255,255,0.05)' }}
              onClick={() => {
                setImage(null);
                setResults(null);
                setInpaintedImage(null);
                setFileName("");
              }}
            >
              Upload Different Image
            </button>
          </div>

          <div className="controls-panel">
            <div className="glass-card">
              <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center' }}>
                <Box size={20} style={{ marginRight: '8px' }} />
                AI Vision Controls
              </h3>

              <div className="control-group" style={{ marginBottom: '1.5rem' }}>
                <label>Detection Queries</label>
                <textarea 
                  rows="2"
                  value={queries}
                  onChange={(e) => setQueries(e.target.value)}
                  placeholder="e.g. girl, dress, trees"
                  style={{ width: '100%', background: 'rgba(0,0,0,0.2)', color: 'white', border: '1px solid var(--glass-border)', borderRadius: '8px', padding: '0.5rem', fontSize: '0.85rem' }}
                />
                <div style={{ marginTop: '0.5rem' }}>
                  <label style={{ fontSize: '0.75rem' }}>Threshold: {threshold}</label>
                  <input 
                    type="range" min="0.01" max="1.0" step="0.01" 
                    value={threshold} 
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    style={{ width: '100%', accentColor: 'var(--accent)' }}
                  />
                </div>
                <button 
                  className="btn btn-secondary" 
                  style={{ width: '100%', marginTop: '0.5rem', fontSize: '0.8rem' }}
                  onClick={() => fileInputRef.current.dispatchEvent(new Event('change', { bubbles: true }))}
                  disabled={!image || loading}
                >
                  Apply & Re-scan
                </button>
              </div>

              <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-dim)', fontSize: '0.9rem' }}>Detected Objects</h4>
              <div className="detection-list">
                {results?.detections?.map((det, index) => (
                  <div 
                    key={index} 
                    className={`detection-item ${selectedObjectIndex === index ? 'selected' : ''}`}
                    onClick={() => {
                      setSelectedObjectIndex(index);
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
                  <h4 style={{ marginTop: '1rem' }}>Inpaint "{results.detections[selectedObjectIndex].label}"</h4>
                  
                  <div className="control-group">
                    <label>Prompt (What to add)</label>
                    <input 
                      type="text" 
                      placeholder="e.g. 'vibrant red silk dress'"
                      value={inpaintPrompt}
                      onChange={(e) => setInpaintPrompt(e.target.value)}
                    />
                  </div>

                  <div className="control-group">
                    <label>Negative Prompt (What to keep/avoid)</label>
                    <input 
                      type="text" 
                      value={negativePrompt}
                      onChange={(e) => setNegativePrompt(e.target.value)}
                    />
                  </div>

                  <div className="control-group">
                    <label>Transformation Strength: {strength}</label>
                    <input 
                      type="range" 
                      min="0.1" max="1.0" step="0.05"
                      value={strength}
                      onChange={(e) => setStrength(parseFloat(e.target.value))}
                      style={{ width: '100%', accentColor: 'var(--accent)' }}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem' }}>
                      <span>Keep Shape (0.1)</span>
                      <span>Total Change (1.0)</span>
                    </div>
                  </div>

                  <button className="btn btn-primary" onClick={handleInpaint} disabled={!inpaintPrompt || inpainting}>
                    <Sparkles size={18} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                    {inpainting ? 'Processing...' : 'Run Inpainting'}
                  </button>
                </div>
              )}

              {inpaintedImage && (
                <div style={{ marginTop: '1.5rem', textAlign: 'center' }}>
                  <p style={{ color: 'var(--accent)', fontWeight: '600', marginBottom: '1rem' }}>✨ Scene Transformation Complete!</p>
                  <button className="btn btn-primary" style={{ width: '100%' }} onClick={() => {
                    setImage(null);
                    setResults(null);
                    setInpaintedImage(null);
                    setFileName("");
                  }}>
                    Explore Another Image
                  </button>
                </div>
              )}
            </div>
            
            <div className="glass-card" style={{ marginTop: '1rem', fontSize: '0.9rem', color: 'var(--text-dim)' }}>
               <h4 style={{ color: 'white', marginBottom: '0.5rem' }}>System Status</h4>
               <ul style={{ listStyle: 'none', padding: 0 }}>
                 <li style={{ color: systemStatus.backend === 'active' ? 'var(--accent)' : 'var(--text-dim)' }}>
                   {systemStatus.backend === 'active' ? '✓' : '○'} Backend Server: {systemStatus.backend}
                 </li>
                 <li style={{ color: systemStatus.agents === 'ready' ? 'var(--accent)' : 'var(--text-dim)' }}>
                   {systemStatus.agents === 'ready' ? '✓' : '○'} AI Specialist Agents: {systemStatus.agents}
                 </li>
                 <li style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
                   Mode: Multi-Agent Parallel Orchestration
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
