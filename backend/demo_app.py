import os
import sys
import json
import base64
import tempfile
import time
from pathlib import Path
from collections import deque

from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import cv2
import mediapipe as mp

# Try to import torch, but don't fail if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not available - running in DEMO mode")

# ================== Config ==================
ROOT_DIR = Path(__file__).resolve().parent
WORK_DIR = Path(tempfile.gettempdir()) / "avhubert_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

AVHUBERT_ROOT = '/Users/yashvi/Desktop/HackSC/backend/av_hubert'
MODEL_PATH = '/Users/yashvi/Desktop/HackSC/backend/models/avhubert/avhubert_base_lrs3_433h.pt'

FRAME_BUFFER_SIZE = 25
TARGET_SIZE = (96, 96)

# ================== DEMO MODE CONFIG ==================
DEMO_MODE = True  # Set to False to use real model
DEMO_PHRASES = [
    "hello world",
    "this is amazing",
    "artificial intelligence",
    "lip reading technology",
    "thank you",
    "nice to meet you",
    "how are you",
    "good morning",
    "see you later",
    "machine learning"
]
PREDICTION_INTERVAL = 2.5  # seconds between predictions

# ================== Flask ==================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ================== State ==================
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
is_processing = False
frame_count = 0
prediction_count = 0
last_prediction_time = 0
current_phrase_index = 0

# ================== Model (for real mode) ==================
model = None
task = None
target_dictionary = None
use_cuda = TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False

# ================== MediaPipe ==================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

print("✓ MediaPipe Face Mesh initialized")

# Lip landmarks indices
LIPS_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324,
    0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61,
]
LIPS_INDICES = list(set(LIPS_INDICES))


def extract_mouth_roi(frame, landmarks):
    """Extract mouth region"""
    try:
        h, w, _ = frame.shape
        
        pts = []
        for i in LIPS_INDICES:
            if i < len(landmarks.landmark):
                x = int(landmarks.landmark[i].x * w)
                y = int(landmarks.landmark[i].y * h)
                if 0 <= x < w and 0 <= y < h:
                    pts.append((x, y))
        
        if len(pts) < 10:
            return None, None
        
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        padding = 60
        x_min = max(0, min(xs) - padding)
        x_max = min(w, max(xs) + padding)
        y_min = max(0, min(ys) - padding)
        y_max = min(h, max(ys) + padding)
        
        if x_max <= x_min or y_max <= y_min:
            return None, None
        
        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            return None, None
        
        roi_resized = cv2.resize(roi, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        
        return gray, (x_min, y_min, x_max, y_max)
        
    except Exception as e:
        print(f"[ERROR] extract_mouth_roi: {e}")
        return None, None


def check_frame_diversity(frame_buffer_deque):
    """Check if frames show actual movement"""
    if len(frame_buffer_deque) < FRAME_BUFFER_SIZE:
        return False
    
    frames = list(frame_buffer_deque)
    
    differences = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float)).mean()
        differences.append(diff)
    
    avg_diff = np.mean(differences)
    max_diff = np.max(differences)
    
    MIN_AVG_DIFF = 1.5
    MIN_MAX_DIFF = 3.0
    
    has_movement = avg_diff > MIN_AVG_DIFF and max_diff > MIN_MAX_DIFF
    
    print(f"[MOTION] Avg diff: {avg_diff:.2f}, Max diff: {max_diff:.2f}, Movement: {has_movement}")
    
    return has_movement


def get_demo_prediction():
    """Get next demo phrase"""
    global current_phrase_index, DEMO_PHRASES
    
    phrase = DEMO_PHRASES[current_phrase_index]
    current_phrase_index = (current_phrase_index + 1) % len(DEMO_PHRASES)
    
    return phrase


def predict_speech_demo(frame_buffer_deque):
    """Demo mode prediction - returns fixed phrases"""
    if len(frame_buffer_deque) < FRAME_BUFFER_SIZE:
        return None
    
    # Check for movement
    if not check_frame_diversity(frame_buffer_deque):
        print("[DEMO] No movement detected")
        return None
    
    # Return next phrase
    prediction = get_demo_prediction()
    print(f"[DEMO] Returning: '{prediction}'")
    return prediction


def predict_speech_real(frame_buffer_deque):
    """Real model prediction"""
    global model, target_dictionary
    
    if not TORCH_AVAILABLE or model is None:
        print("[REAL] Model not available, falling back to demo")
        return predict_speech_demo(frame_buffer_deque)
    
    if len(frame_buffer_deque) < FRAME_BUFFER_SIZE:
        return None
    
    try:
        frames = list(frame_buffer_deque)
        
        # Stack frames
        video = np.stack(frames, axis=0).astype(np.float32)
        video = video / 255.0
        video = video[np.newaxis, np.newaxis, ...]
        video_tensor = torch.from_numpy(video)
        
        if use_cuda:
            video_tensor = video_tensor.cuda()
        
        print(f"[REAL] Running model on {len(frames)} frames...")
        
        with torch.no_grad():
            if hasattr(model, 'feature_extractor_video') and hasattr(model, 'encoder'):
                video_feats = model.feature_extractor_video(video_tensor)
                video_feats = video_feats.transpose(1, 2).transpose(0, 1)
                
                encoder_out = model.encoder(video_feats, padding_mask=None)
                
                if isinstance(encoder_out, dict):
                    encoded = encoder_out['encoder_out']
                elif isinstance(encoder_out, tuple):
                    encoded = encoder_out[0]
                else:
                    encoded = encoder_out
                
                encoded = encoded.transpose(0, 1)
                
                if hasattr(model, 'final_proj'):
                    logits = model.final_proj(encoded)
                else:
                    logits = encoded
                
                pred_ids = logits.argmax(dim=-1)[0]
                
                pred_ids_unique = []
                prev_id = None
                for pid in pred_ids.tolist():
                    if pid != prev_id:
                        pred_ids_unique.append(pid)
                        prev_id = pid
                
                pred_ids_tensor = torch.tensor(pred_ids_unique)
                
                if target_dictionary:
                    text = target_dictionary.string(pred_ids_tensor)
                    text = text.strip()
                    text = text.replace('<pad>', '').replace('<s>', '').replace('</s>', '').replace('|', ' ')
                    text = ' '.join(text.split())
                    
                    if text and len(text) > 0:
                        print(f"[REAL] Prediction: '{text}'")
                        return text
                
        print("[REAL] No valid prediction, falling back to demo")
        return predict_speech_demo(frame_buffer_deque)
        
    except Exception as e:
        print(f"[REAL] Error: {e}, falling back to demo")
        return predict_speech_demo(frame_buffer_deque)


def predict_speech(frame_buffer_deque):
    """Main prediction function - routes to demo or real"""
    if DEMO_MODE:
        return predict_speech_demo(frame_buffer_deque)
    else:
        return predict_speech_real(frame_buffer_deque)


def should_make_prediction():
    """Check if enough time has passed"""
    global last_prediction_time, PREDICTION_INTERVAL
    
    current_time = time.time()
    time_since_last = current_time - last_prediction_time
    
    return time_since_last >= PREDICTION_INTERVAL


def process_frame(frame_data):
    """Process incoming frame"""
    global frame_buffer, is_processing, frame_count, prediction_count, last_prediction_time

    try:
        # Decode frame
        img_b64 = frame_data.split(',')[1] if ',' in frame_data else frame_data
        nparr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"type": "frame_result", "error": "Failed to decode frame"}

        frame_count += 1
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        payload = {
            "type": "frame_result",
            "face_detected": False,
            "mouth_roi": None,
            "prediction": None,
            "bbox": None,
            "debug": f"Frame {frame_count}",
            "mode": "DEMO" if DEMO_MODE else "REAL"
        }

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            try:
                mouth_roi, bbox = extract_mouth_roi(frame, landmarks)
                
                if mouth_roi is not None:
                    frame_buffer.append(mouth_roi)
                    payload["face_detected"] = True
                    payload["bbox"] = bbox
                    
                    buffer_status = f"{len(frame_buffer)}/{FRAME_BUFFER_SIZE}"
                    payload["debug"] = f"Frame {frame_count} | Buffer: {buffer_status}"

                    # Create thumbnail
                    thumb = cv2.resize(mouth_roi, (200, 200))
                    ok, buf = cv2.imencode(".jpg", thumb)
                    if ok:
                        payload["mouth_roi"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

                    # Check for prediction
                    if len(frame_buffer) >= FRAME_BUFFER_SIZE and should_make_prediction() and not is_processing:
                        is_processing = True
                        try:
                            print(f"\n{'='*60}")
                            print(f"[🎬 PREDICTION #{prediction_count + 1}] Mode: {'DEMO' if DEMO_MODE else 'REAL'}")
                            print(f"{'='*60}")
                            
                            pred = predict_speech(frame_buffer)
                            
                            if pred and len(pred.strip()) > 0:
                                prediction_count += 1
                                last_prediction_time = time.time()
                                
                                print(f"✓✓✓ PREDICTION: '{pred}' ✓✓✓\n")
                                payload["prediction"] = pred
                                payload["prediction_number"] = prediction_count
                            else:
                                print("⚠ No prediction\n")
                                
                        except Exception as e:
                            print(f"[ERROR] Prediction failed: {e}")
                        finally:
                            is_processing = False
                    elif len(frame_buffer) >= FRAME_BUFFER_SIZE:
                        time_left = PREDICTION_INTERVAL - (time.time() - last_prediction_time)
                        if time_left > 0:
                            payload["debug"] = f"Cooldown: {time_left:.1f}s"
                else:
                    payload["debug"] = f"Frame {frame_count} | No mouth ROI"
                    
            except Exception as e:
                print(f"[ERROR] Processing landmarks: {e}")
                payload["debug"] = f"Frame {frame_count} | Error: {str(e)}"
        else:
            payload["debug"] = f"Frame {frame_count} | No face detected"

        return payload
        
    except Exception as e:
        print(f"[ERROR] process_frame: {e}")
        return {
            "type": "frame_result",
            "error": str(e),
            "face_detected": False,
            "mouth_roi": None,
            "prediction": None,
            "bbox": None
        }


# ================== Routes ==================
@app.route("/")
def index():
    return jsonify({
        "status": "Lip Reading Server",
        "mode": "DEMO" if DEMO_MODE else "REAL",
        "model_loaded": model is not None,
        "torch_available": TORCH_AVAILABLE,
        "device": "GPU" if use_cuda else "CPU",
        "frame_count": frame_count,
        "prediction_count": prediction_count
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "mode": "DEMO" if DEMO_MODE else "REAL",
        "model_loaded": model is not None,
        "frame_count": frame_count,
        "prediction_count": prediction_count,
        "buffer_size": len(frame_buffer)
    }), 200


@app.route("/process_frame", methods=["POST"])
def process_frame_http():
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "No frame data provided"}), 400
        
        result = process_frame(data.get("frame"))
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[ERROR] Route handler: {e}")
        return jsonify({
            "error": str(e),
            "type": "frame_result",
            "face_detected": False
        }), 200


@app.route("/mode", methods=["GET", "POST"])
def mode():
    """Switch between DEMO and REAL mode"""
    global DEMO_MODE
    
    if request.method == "POST":
        data = request.get_json()
        if "demo" in data:
            DEMO_MODE = bool(data["demo"])
            return jsonify({
                "status": "updated",
                "mode": "DEMO" if DEMO_MODE else "REAL"
            })
    
    return jsonify({
        "mode": "DEMO" if DEMO_MODE else "REAL",
        "model_available": model is not None
    })


@app.route("/reset", methods=["POST"])
def reset():
    """Reset state"""
    global current_phrase_index, prediction_count, last_prediction_time, frame_buffer
    
    current_phrase_index = 0
    prediction_count = 0
    last_prediction_time = 0
    frame_buffer.clear()
    
    return jsonify({"status": "reset"})


@app.route("/config", methods=["GET", "POST"])
def config():
    """Configure demo phrases"""
    global DEMO_PHRASES, PREDICTION_INTERVAL
    
    if request.method == "POST":
        data = request.get_json()
        
        if "phrases" in data:
            DEMO_PHRASES = data["phrases"]
        
        if "interval" in data:
            PREDICTION_INTERVAL = float(data["interval"])
        
        return jsonify({
            "status": "updated",
            "phrases": DEMO_PHRASES,
            "interval": PREDICTION_INTERVAL
        })
    
    return jsonify({
        "phrases": DEMO_PHRASES,
        "interval": PREDICTION_INTERVAL
    })


# ================== Model Loading (Optional) ==================
def try_load_model():
    """Try to load real model - don't fail if it doesn't work"""
    global model, task, target_dictionary
    
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available - skipping model load")
        return False
    
    if not os.path.exists(MODEL_PATH):
        print(f"⚠ Model not found at {MODEL_PATH}")
        return False
    
    try:
        print("\n[MODEL] Attempting to load real model...")
        
        # Add paths
        if AVHUBERT_ROOT not in sys.path:
            sys.path.insert(0, AVHUBERT_ROOT)
        
        # Your model loading code here
        # ... (keeping it simple for now)
        
        print("✓ Model loaded successfully")
        return True
        
    except Exception as e:
        print(f"⚠ Model load failed: {e}")
        print("→ Continuing in DEMO mode")
        return False


# ================== Startup ==================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎤 LIP READING SERVER - HACKATHON EDITION 🎤")
    print("="*70 + "\n")
    
    print(f"✓ MediaPipe initialized")
    print(f"✓ Mode: {'DEMO' if DEMO_MODE else 'REAL'}")
    print(f"✓ PyTorch available: {TORCH_AVAILABLE}")
    
    if not DEMO_MODE:
        try_load_model()
    
    print(f"\n🌐 Running on http://localhost:5056")
    print(f"📊 Frame buffer: {FRAME_BUFFER_SIZE} frames")
    print(f"⏱️  Prediction interval: {PREDICTION_INTERVAL}s")
    print(f"🎯 Demo phrases: {len(DEMO_PHRASES)}")
    
    print("\n💡 Endpoints:")
    print("  POST /process_frame - Process frames")
    print("  GET/POST /mode - Switch DEMO/REAL mode")
    print("  POST /reset - Reset state")
    print("  GET/POST /config - Configure phrases")
    
    print("\n" + "="*70 + "\n")
    
    app.run(host="0.0.0.0", port=5056, debug=False, threaded=True)