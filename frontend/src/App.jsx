import React, { useState, useEffect, useRef, useCallback } from 'react';

/* ─── MediaPipe globals (loaded via CDN in index.html) ────────────── */
const MP = () => window;

function App() {
  // ── state ──
  const [streaming, setStreaming]   = useState(false);
  const [calibrated, setCalibrated] = useState(false);
  const [connected, setConnected]   = useState(false);
  const [overallFeedback, setOverallFeedback] = useState([]);
  const [scores, setScores]         = useState({ rom_score: 0, stability_score: 0, quality_score: 0, total_score: 0 });
  const [metrics, setMetrics]       = useState({ maxShoulder: 0, maxTrunk: 0, isCompensatory: false, mlConf: 0 });
  const [loggedIn, setLoggedIn]     = useState('');
  const [token, setToken]           = useState('');

  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const wsRef     = useRef(null);
  const poseRef   = useRef(null);
  const camRef    = useRef(null);
  const frameId   = useRef(0);

  // ── WebSocket ──
  const connectWS = useCallback((authToken) => {
    const ws = new WebSocket('ws://localhost:8000/ws/stream');
    ws.onopen = () => {
      setConnected(true);
      // send auth handshake
      ws.send(JSON.stringify({ token: authToken || '' }));
    };
    ws.onclose = () => setConnected(false);

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);

      if (msg.type === 'auth') {
        if (msg.username && msg.username !== 'guest') {
          setLoggedIn(msg.username);
        }
        return;
      }

      if (msg.type === 'frame' && msg.features) {
        const f = msg.features;
        setCalibrated(msg.is_calibrated);

        const pt = {
          shoulder: Math.round((f.right_shoulder_angle || 0) * 10) / 10,
          trunk:    Math.round((f.trunk_lean || f.absolute_trunk_lean || 0) * 10) / 10,
        };

        setMetrics(prev => ({
          maxShoulder:    Math.max(prev.maxShoulder, pt.shoulder),
          maxTrunk:       Math.max(prev.maxTrunk, pt.trunk),
          isCompensatory: msg.is_compensatory,
          mlConf:         msg.ml_confidence || 0,
        }));

        if (msg.scores) setScores(msg.scores);

        if (msg.overall_feedback) {
          setOverallFeedback(msg.overall_feedback);
        }
      }
    };

    wsRef.current = ws;
  }, []);

  // ── MediaPipe Pose ──
  const startCamera = useCallback(() => {
    const { Pose, Camera, drawConnectors, drawLandmarks, POSE_CONNECTIONS } = MP();

    if (!Pose) {
      alert('MediaPipe not loaded yet — please refresh the page.');
      return;
    }

    const pose = new Pose({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${f}`,
    });

    pose.setOptions({
      modelComplexity: 0,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults((results) => {
      // draw skeleton overlay
      const ctx = canvasRef.current?.getContext('2d');
      if (ctx && canvasRef.current) {
        const w = canvasRef.current.width  = videoRef.current.videoWidth;
        const h = canvasRef.current.height = videoRef.current.videoHeight;
        ctx.clearRect(0, 0, w, h);

        if (results.poseLandmarks) {
          drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS,
            { color: 'rgba(34,211,238,.45)', lineWidth: 2 });
          drawLandmarks(ctx, results.poseLandmarks,
            { color: '#22d3ee', lineWidth: 1, radius: 3 });

          // send every 2nd frame to backend
          frameId.current += 1;
          if (frameId.current % 2 === 0 && wsRef.current?.readyState === 1) {
            const landmarks = results.poseLandmarks.map(l => ({
              x: l.x, y: l.y, z: l.z, visibility: l.visibility,
            }));
            wsRef.current.send(JSON.stringify({ landmarks }));
          }
        }
      }
    });

    poseRef.current = pose;

    const cam = new Camera(videoRef.current, {
      onFrame: async () => { await pose.send({ image: videoRef.current }); },
      width: 1280,
      height: 720,
    });
    cam.start();
    camRef.current = cam;
    setStreaming(true);
  }, []);

  const stopCamera = useCallback(() => {
    camRef.current?.stop();
    poseRef.current?.close();
    wsRef.current?.close();
    setStreaming(false);
    setConnected(false);
    setCalibrated(false);
    setOverallFeedback([]);
    setMetrics({ maxShoulder: 0, maxTrunk: 0, isCompensatory: false, mlConf: 0 });
    setScores({ rom_score: 0, stability_score: 0, quality_score: 0, total_score: 0 });
  }, []);

  const handleStart = () => {
    connectWS(token);
    startCamera();
  };


  // cleanup on unmount
  useEffect(() => () => { stopCamera(); }, [stopCamera]);

  // ── colour helper ──
  const scoreColor = (v) => v >= 75 ? 'green' : v >= 50 ? 'amber' : 'red';

  // ── render ──
  return (
    <div className="app-shell">
      {/* ── Top bar ── */}
      <header className="top-bar">
        <h1>AI Rehab · Motion Analysis</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {loggedIn && <span className="status-pill">👤 {loggedIn}</span>}
          <span className="status-pill">
            <span className={`dot ${connected ? 'live' : ''}`} />
            {connected ? 'Live' : 'Offline'}
          </span>
        </div>
      </header>

      {/* ── Main grid ── */}
      <div className="main-grid">
        {/* LEFT: camera */}
        <div className="camera-panel">
          <div className="camera-feed" id="camera-feed">
            <video ref={videoRef} playsInline muted
              style={{ display: streaming ? 'block' : 'none' }} />
            <canvas ref={canvasRef} />

            {streaming && !calibrated && (
              <div className="calibration-overlay">
                <div className="spinner" />
                <p>Calibrating baseline posture…</p>
                <span style={{ fontSize: '.8rem', color: '#94a3b8' }}>
                  Stand still facing the camera
                </span>
              </div>
            )}

            {!streaming && (
              <div className="empty-state" style={{ height: '100%' }}>
                <span className="icon">📷</span>
                <p>Camera feed will appear here</p>
              </div>
            )}
          </div>

          <div className="cam-controls">
            {!streaming ? (
              <button className="btn btn-primary" onClick={handleStart} id="btn-start">
                ▶ Start Session
              </button>
            ) : (
              <button className="btn btn-danger" onClick={stopCamera} id="btn-stop">
                ■ Stop Session
              </button>
            )}
            <span className={`status-badge ${metrics.isCompensatory ? 'compensatory' : 'healthy'}`}>
              {metrics.isCompensatory ? '⚠ Compensation Detected' : '✓ Healthy Movement'}
            </span>
          </div>
        </div>

        {/* RIGHT: dashboard */}
        <div className="dashboard-col">
          {/* Clinical scores */}
          <div className="scores-row">
            {[
              { label: 'ROM Score',       key: 'rom_score' },
              { label: 'Stability',       key: 'stability_score' },
              { label: 'Quality',         key: 'quality_score' },
              { label: 'Overall Score',   key: 'total_score' },
            ].map(({ label, key }) => (
              <div className="score-card" key={key} id={`score-${key}`}>
                <span className="label">{label}</span>
                <span className={`value ${scoreColor(scores[key])}`}>
                  {scores[key]}
                </span>
                <span className="sub">/ 100</span>
              </div>
            ))}
          </div>

          {/* Live metrics */}
          <div className="metrics-row">
            <div className="metric-card" id="metric-shoulder">
              <div className="metric-label">Max Shoulder ROM</div>
              <div className="metric-value">{metrics.maxShoulder}°</div>
            </div>
            <div className="metric-card" id="metric-trunk">
              <div className="metric-label">Max Trunk Lean</div>
              <div className="metric-value"
                style={{ color: metrics.maxTrunk > 15 ? '#f87171' : undefined }}>
                {metrics.maxTrunk}°
              </div>
            </div>
            <div className="metric-card" id="metric-ml">
              <div className="metric-label">ML Confidence</div>
              <div className="metric-value"
                style={{ color: metrics.mlConf > 0.5 ? '#f87171' : '#34d399' }}>
                {(metrics.mlConf * 100).toFixed(0)}%
              </div>
            </div>
          </div>


          {/* Posture Recommendations */}
          <div className="feedback-panel" id="feedback-panel">
            <h3><span>📋</span> Posture Recommendations</h3>
            <p className="subtitle">Overall analysis & corrective guidance</p>
            {overallFeedback.length > 0 ? (
              <div className="feedback-grid">
                {overallFeedback.map((item, i) => (
                  <div key={i} className={`feedback-card ${item.status}`}>
                    <div className="card-header">
                      <span>{item.category}</span>
                      <span className="status-tag">
                        {item.status === 'excellent' ? '✓ Excellent' : '⚠ Action Required'}
                      </span>
                    </div>
                    <div className="message">{item.message}</div>
                    <div className="tip">{item.tip}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state" style={{ padding: '2rem 1rem' }}>
                <span className="icon">📊</span>
                <p>Recommendations will populate as you begin movement</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
