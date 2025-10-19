import os
import sys

# ✅ FIX FOR MAC: Disable TensorFlow Metal/MPS before imports
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN

# ✅ CRITICAL: Set these BEFORE importing anything
import warnings
warnings.filterwarnings('ignore')
 #✅ NEW: Disable MPS/Metal completely
os.environ['DISABLE_MPS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# ✅ NEW: Force TensorFlow to use legacy Keras
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import json
import base64
import tempfile
import random
from pathlib import Path
from collections import deque
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import cv2
import mediapipe as mp
import torch

import time

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# ================== NEW: Session startup tracking ==================
SESSION_START_TIME = time.time()
STARTUP_GRACE_PERIOD = 5.0  # 5 seconds delay before template matching
# ===================================================================


# ================== EMOTION DETECTION - DEEPFACE ==================


EMOTION_DETECTION_AVAILABLE = False
emotion_detector = None

last_frame_time = 0
MIN_FRAME_INTERVAL = 0.033 

print("\n" + "="*70)
print("🔍 INITIALIZING EMOTION DETECTION (DeepFace)")
print("="*70)

try:
    print("[1/2] Importing DeepFace...")
    
    # ✅ Force CPU backend for DeepFace on Mac
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
    
    from deepface import DeepFace
    print("      ✓ DeepFace imported successfully")
    
    print("[2/2] DeepFace ready for emotion detection (CPU mode)...")
    EMOTION_DETECTION_AVAILABLE = True
    emotion_detector = "deepface"
    
    print("\n" + "="*70)
    print("✓✓✓ EMOTION DETECTION: ENABLED (DeepFace CPU) ✓✓✓")
    print("="*70 + "\n")
    
except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    print("   Solution: pip install deepface tf-keras")
    EMOTION_DETECTION_AVAILABLE = False
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    EMOTION_DETECTION_AVAILABLE = False

print("="*70)
print(f"FINAL STATUS: EMOTION_DETECTION_AVAILABLE = {EMOTION_DETECTION_AVAILABLE}")
print("="*70 + "\n")

# ================== PREDICTION MODE TOGGLE ==================
USE_MOCK_PREDICTIONS = False  # Set to False to use real model
USE_TEMPLATE_MATCHING = True  # ✅ NEW: Set to True for template matching
# ===========================================================

# Fixed vocabulary for mock predictions
MOCK_VOCABULARY = [
    "hello", "yes", "no", "please", "thank you",
    "help", "stop", "go", "wait", "good", "bad",
    "morning", "evening", "water", "food", "eat", "drink",
    "speak", "listen", "understand", "sorry", "excuse me",
    "how", "what", "when", "where", "who", "why"
]

EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': os.environ.get('SENDER_EMAIL', 'your-email@gmail.com'),
    'sender_password': os.environ.get('EMAIL_PASSWORD', 'your-app-password'),
    'sender_name': 'Echoes Report'
}

RECORDING_MODE = False
RECORDING_WORD = None
recorded_frames = []

# ================== TEMPLATE MATCHING SYSTEM ==================
import glob
from sklearn.metrics.pairwise import cosine_similarity

# Load all templates at startup
template_library = {}

def load_all_templates():
    """Load all saved templates from disk"""
    global template_library
    
    template_dir = "recorded_templates"
    if not os.path.exists(template_dir):
        print(f"[TEMPLATES] No template directory found at {template_dir}")
        return
    
    template_files = glob.glob(f"{template_dir}/*.npy")
    
    if len(template_files) == 0:
        print(f"[TEMPLATES] No templates found in {template_dir}")
        return
    
    for filepath in template_files:
        filename = os.path.basename(filepath)
        # Format: word_timestamp.npy
        word = filename.split('_')[0]
        
        try:
            frames = np.load(filepath)
            
            if word not in template_library:
                template_library[word] = []
            
            template_library[word].append({
                'frames': frames,
                'filepath': filepath,
                'timestamp': filename.split('_')[1].replace('.npy', '')
            })
            
        except Exception as e:
            print(f"[ERROR] Failed to load {filepath}: {e}")
    
    print(f"\n{'='*70}")
    print(f"[TEMPLATES] Loaded {sum(len(v) for v in template_library.values())} templates for {len(template_library)} words")
    for word, templates in template_library.items():
        print(f"  - '{word}': {len(templates)} template(s)")
    print(f"{'='*70}\n")

def compute_sequence_similarity(seq1, seq2):
    """
    Compute similarity between two sequences of frames using DTW-like approach
    
    Args:
        seq1: numpy array (T1, H, W)
        seq2: numpy array (T2, H, W)
    
    Returns:
        similarity score (0-1)
    """
    try:
        # Flatten frames to vectors
        seq1_flat = seq1.reshape(seq1.shape[0], -1)  # (T1, H*W)
        seq2_flat = seq2.reshape(seq2.shape[0], -1)  # (T2, H*W)
        
        # Normalize
        seq1_norm = seq1_flat / (np.linalg.norm(seq1_flat, axis=1, keepdims=True) + 1e-8)
        seq2_norm = seq2_flat / (np.linalg.norm(seq2_flat, axis=1, keepdims=True) + 1e-8)
        
        # Simple approach: Compare average frame representations
        seq1_avg = seq1_norm.mean(axis=0)
        seq2_avg = seq2_norm.mean(axis=0)
        
        similarity = cosine_similarity([seq1_avg], [seq2_avg])[0][0]
        
        return max(0, similarity)  # Clamp to [0, 1]
        
    except Exception as e:
        print(f"[ERROR] Similarity computation: {e}")
        return 0.0

def predict_using_templates(frame_buffer_deque):
    """
    Predict word by matching against templates using cosine similarity
    ✅ ENHANCED: Checks for startup grace period
    
    Args:
        frame_buffer_deque: deque of mouth ROI frames
    
    Returns:
        predicted word or None
    """
    # ✅ CHECK: Don't run during startup grace period
    time_since_startup = time.time() - SESSION_START_TIME
    if time_since_startup < STARTUP_GRACE_PERIOD:
        remaining = STARTUP_GRACE_PERIOD - time_since_startup
        print(f"[TEMPLATE MATCHING] ⏳ Waiting for startup grace period ({remaining:.1f}s remaining)")
        return None
    
    if len(frame_buffer_deque) < FRAME_BUFFER_SIZE:
        return None
    
    if len(template_library) == 0:
        print("[TEMPLATE MATCHING] No templates loaded. Record some first!")
        return None
    
    try:
        # Convert buffer to numpy array
        input_frames = np.array(list(frame_buffer_deque))
        
        print(f"[TEMPLATE MATCHING] Comparing against {len(template_library)} words...")
        
        # Compare against all templates
        best_word = None
        best_score = -1
        all_scores = {}
        
        for word, templates in template_library.items():
            word_scores = []
            
            for template in templates:
                template_frames = template['frames']
                
                # Resize input to match template length if needed
                if len(input_frames) != len(template_frames):
                    # Simple interpolation: sample frames evenly
                    indices = np.linspace(0, len(input_frames)-1, len(template_frames)).astype(int)
                    input_resampled = input_frames[indices]
                else:
                    input_resampled = input_frames
                
                # Compute similarity
                score = compute_sequence_similarity(input_resampled, template_frames)
                word_scores.append(score)
            
            # Average score across all templates for this word
            avg_score = np.mean(word_scores)
            all_scores[word] = avg_score
            
            if avg_score > best_score:
                best_score = avg_score
                best_word = word
        
        # Print all scores
        print(f"[TEMPLATE MATCHING] Scores:")
        for word, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {word}: {score:.3f}")
        
        # Confidence threshold
        MIN_CONFIDENCE = 0.5  # Adjust this threshold
        
        if best_score >= MIN_CONFIDENCE:
            print(f"[TEMPLATE MATCHING] ✓ Matched: '{best_word}' (confidence: {best_score:.3f})")
            return best_word
        else:
            print(f"[TEMPLATE MATCHING] ✗ No confident match (best: {best_score:.3f}, need: {MIN_CONFIDENCE})")
            return None
            
    except Exception as e:
        print(f"[ERROR] Template matching: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================

def start_recording(word):
    """Start recording a template for a word"""
    global RECORDING_MODE, RECORDING_WORD, recorded_frames
    RECORDING_MODE = True
    RECORDING_WORD = word
    recorded_frames = []
    print(f"\n{'='*60}")
    print(f"🔴 RECORDING STARTED for word: '{word}'")
    print(f"Speak the word now!")
    print(f"{'='*60}\n")

def stop_recording():
    """Stop recording and save template safely"""
    global RECORDING_MODE, RECORDING_WORD, recorded_frames

    if not RECORDING_MODE:
        return False, "Not recording"

    RECORDING_MODE = False

    if len(recorded_frames) < 30:
        RECORDING_WORD = None
        recorded_frames = []
        return False, f"Not enough frames ({len(recorded_frames)}), need at least 30"

    save_dir = "recorded_templates"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = int(time.time())
    filename = f"{save_dir}/{RECORDING_WORD}_{timestamp}.npy"

    # Ensure consistent dtype
    np.save(filename, np.array(recorded_frames, dtype=np.uint8))

    print(f"\n{'='*60}")
    print(f"✓ Saved {len(recorded_frames)} frames to {filename}")
    print(f"{'='*60}\n")

    word = RECORDING_WORD
    RECORDING_WORD = None
    recorded_frames = []

    return True, f"Saved {filename}"

def generate_html_report(report_data, session_id=None):
    """
    Generate HTML email content for the session report
    
    Args:
        report_data: Dictionary with session information
        session_id: Optional session ID
    
    Returns:
        str: HTML content
    """
    predictions = report_data.get('predictions', [])
    emotions = report_data.get('emotions', [])
    #session_duration = report_data.get('duration', 'N/A')
    
    
    emotion_counts = {}
    for emotion in emotions:
        # Handle both dict and string formats
        if isinstance(emotion, dict):
            emotion_name = emotion.get('emotion', 'neutral')
        else:
            emotion_name = emotion
        
        emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
        
    # Build predictions list
    predictions_html = ""
    for i, pred in enumerate(predictions, 1):
        word = pred.get('word', 'N/A')
        timestamp = pred.get('timestamp', 'N/A')
        confidence = pred.get('confidence', 0)
        
        predictions_html += f"""
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">{i}</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: 600;">{word}</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">{timestamp}</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">{confidence}%</td>
        </tr>
        """
    
    if not predictions_html:
        predictions_html = """
        <tr>
            <td colspan="4" style="padding: 20px; text-align: center; color: #999;">
                No predictions recorded in this session
            </td>
        </tr>
        """
    
    # Build emotion summary
    emotion_summary_html = ""
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(emotions) * 100) if emotions else 0
        emotion_summary_html += f"""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: 500; text-transform: capitalize;">{emotion}</span>
                <span>{count} times ({percentage:.1f}%)</span>
            </div>
            <div style="background: #f0f0f0; height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="background: #4CAF50; height: 100%; width: {percentage}%;"></div>
            </div>
        </div>
        """
    
    if not emotion_summary_html:
        emotion_summary_html = """
        <p style="color: #999; text-align: center; padding: 20px;">
            No emotion data recorded
        </p>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f5f5f5;">
        <div style="max-width: 600px; margin: 0 auto; background: white; padding: 0;">
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; text-align: center;">
                <h1 style="margin: 0; color: white; font-size: 28px; font-weight: 600;">
                    Echoes: Digital Report for Yashvi
                </h1>
                <p style="margin: 10px 0 0 0; color: rgba(255,255,255,0.9); font-size: 14px;">
                    Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                </p>
                {f'<p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.8); font-size: 12px;">Session ID: {session_id}</p>' if session_id else ''}
            </div>
            
            <!-- Session Summary -->
            <div style="padding: 30px;">
                <h2 style="margin: 0 0 20px 0; color: #333; font-size: 20px;">
                    📊 Session Summary
                </h2>
                
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        
                        <div>
                            <div style="color: #666; font-size: 12px; margin-bottom: 5px;">PREDICTIONS</div>
                            <div style="color: #333; font-size: 18px; font-weight: 600;">{len(predictions)}</div>
                        </div>
                        <div>
                            <div style="color: #666; font-size: 12px; margin-bottom: 5px;">DOMINANT EMOTION</div>
                            <div style="color: #333; font-size: 18px; font-weight: 600; text-transform: capitalize;">{dominant_emotion}</div>
                        </div>
                    </div>
                </div>
                
                <!-- Predictions Table -->
                <h2 style="margin: 0 0 15px 0; color: #333; font-size: 20px;">
                    💬 Detected Words
                </h2>
                
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px;">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e0e0e0; color: #666; font-size: 12px; font-weight: 600;">#</th>
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e0e0e0; color: #666; font-size: 12px; font-weight: 600;">WORD</th>
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e0e0e0; color: #666; font-size: 12px; font-weight: 600;">TIME</th>
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e0e0e0; color: #666; font-size: 12px; font-weight: 600;">CONFIDENCE</th>
                        </tr>
                    </thead>
                    <tbody>
                        {predictions_html}
                    </tbody>
                </table>
                
                <!-- Emotion Analysis -->
                <h2 style="margin: 0 0 15px 0; color: #333; font-size: 20px;">
                    😊 Emotion Analysis
                </h2>
                
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                    {emotion_summary_html}
                </div>
            </div>
            
            <!-- Footer -->
            <div style="background: #f8f9fa; padding: 20px 30px; text-align: center; border-top: 1px solid #e0e0e0;">
                <p style="margin: 0; color: #999; font-size: 12px;">
                    This report was automatically generated by Echoes
                </p>
                <p style="margin: 10px 0 0 0; color: #999; font-size: 12px;">
                    For questions or support, please contact your administrator
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def send_report_email(recipient_email, report_data, session_id=None):
    """
    Send the Echoes report via email
    
    Args:
        recipient_email: Email address to send report to
        report_data: Dictionary containing session data (predictions, emotions, etc.)
        session_id: Optional session identifier
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Echoes Report for Patient 1012 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        msg['From'] = f"{EMAIL_CONFIG['sender_name']} <{EMAIL_CONFIG['sender_email']}>"
        msg['To'] = recipient_email
        
        # Generate HTML report
        html_content = generate_html_report(report_data, session_id)
        
        # Attach HTML content
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Connect to SMTP server and send
        print(f"[EMAIL] Connecting to {EMAIL_CONFIG['smtp_server']}...")
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        
        print(f"[EMAIL] Logging in...")
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        
        print(f"[EMAIL] Sending to {recipient_email}...")
        server.send_message(msg)
        server.quit()
        
        print(f"[EMAIL] ✓ Successfully sent report to {recipient_email}")
        return True, f"Report sent successfully to {recipient_email}"
        
    except smtplib.SMTPAuthenticationError:
        error_msg = "Email authentication failed. Please check credentials."
        print(f"[EMAIL ERROR] {error_msg}")
        return False, error_msg
        
    except smtplib.SMTPException as e:
        error_msg = f"SMTP error: {str(e)}"
        print(f"[EMAIL ERROR] {error_msg}")
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        print(f"[EMAIL ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg
    
# ================== NEW: LIP OPENNESS DETECTION ==================
def are_lips_open(landmarks, threshold=0.02):
    """
    Detect if lips are open by measuring vertical distance between upper and lower lips
    
    Args:
        landmarks: MediaPipe face landmarks
        threshold: Minimum distance ratio to consider lips "open"
    
    Returns:
        bool: True if lips are open, False if closed
    """
    try:
        # Upper lip center: landmark 13
        # Lower lip center: landmark 14
        upper_lip = landmarks.landmark[13]
        lower_lip = landmarks.landmark[14]
        
        # Calculate vertical distance
        lip_distance = abs(upper_lip.y - lower_lip.y)
        
        # Also check horizontal distance of mouth corners for reference
        # Left corner: 61, Right corner: 291
        left_corner = landmarks.landmark[61]
        right_corner = landmarks.landmark[291]
        mouth_width = abs(right_corner.x - left_corner.x)
        
        # Normalize lip distance by mouth width
        normalized_distance = lip_distance / mouth_width if mouth_width > 0 else 0
        
        is_open = normalized_distance > threshold
        
        print(f"[LIP DETECTION] Distance: {normalized_distance:.4f}, Threshold: {threshold}, Open: {is_open}")
        
        return is_open
        
    except Exception as e:
        print(f"[ERROR] Lip detection: {e}")
        return True  # Default to open if detection fails
# ==================================================================

def detect_emotion_from_frame(frame):
    """Detect emotion using DeepFace - MAC COMPATIBLE VERSION"""
    if not EMOTION_DETECTION_AVAILABLE:
        return None, 0
    
    try:
        from deepface import DeepFace
        
        # ✅ Resize frame to reduce processing time and avoid shape issues
        # DeepFace works best with smaller images
        max_size = 640
        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        # ✅ Convert BGR to RGB (DeepFace expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # DeepFace analyze with safer settings
        result = DeepFace.analyze(
            frame_rgb,  # Use RGB frame
            actions=['emotion'], 
            enforce_detection=False,
            silent=True,
            detector_backend='opencv'  # ✅ Use OpenCV (faster, more stable on Mac)
        )
        
        # Handle result
        if isinstance(result, list):
            result = result[0]
        
        dominant_emotion = result['dominant_emotion']
        confidence = result['emotion'][dominant_emotion]
        
        # Map emotions
        emotion_map = {
            'angry': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful',
            'happy': 'happy',
            'sad': 'sad',
            'surprise': 'surprised',
            'neutral': 'neutral'
        }
        
        mapped = emotion_map.get(dominant_emotion, 'neutral')
        print(f"[EMOTION] ✓ {mapped} ({confidence:.0f}%)")
        return mapped, int(confidence)
        
    except Exception as e:
        print(f"[EMOTION ERROR] {e}")
        return None, 0
    
def patch_fairseq():
    """Comprehensive patch for fairseq issues"""
    
    # Patch 1: Fix encoder layer building
    try:
        from fairseq.models.wav2vec import wav2vec2
        
        original_build_encoder_layer = wav2vec2.TransformerEncoder.build_encoder_layer
        
        def patched_build_encoder_layer(self, args):
            """Fixed version that ensures layer is always created"""
            layer = None
            
            # Check if using Conformer
            if getattr(args, 'use_conformer', False):
                try:
                    from fairseq.models.wav2vec.wav2vec2_conformer import ConformerWav2Vec2EncoderLayer
                    layer = ConformerWav2Vec2EncoderLayer(args)
                except (ImportError, AttributeError):
                    print("[PATCH] Conformer not available, using standard transformer")
            
            # Fallback to standard TransformerEncoderLayer
            if layer is None:
                from fairseq.modules import TransformerEncoderLayer
                layer = TransformerEncoderLayer(args)
            
            # Apply FSDP wrapping if needed
            if hasattr(self, 'fsdp_wrap'):
                layer = self.fsdp_wrap(layer)
            elif hasattr(wav2vec2, 'fsdp_wrap'):
                layer = wav2vec2.fsdp_wrap(layer)
            
            return layer
        
        wav2vec2.TransformerEncoder.build_encoder_layer = patched_build_encoder_layer
        print("✓ Encoder layer building patched")
        
    except Exception as e:
        print(f"⚠ Warning: Could not patch encoder: {e}")
    
    # [Rest of patch_fairseq remains the same...]
    
# ================== Config ==================
ROOT_DIR = Path(__file__).resolve().parent
WORK_DIR = Path(tempfile.gettempdir()) / "avhubert_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

AVHUBERT_ROOT = '/Users/yashvi/Desktop/HackSC/backend/av_hubert'
MODEL_PATH = '/Users/yashvi/Desktop/HackSC/backend/models/avhubert/avhubert_base_lrs3_433h.pt'

FRAME_BUFFER_SIZE = 40
TARGET_SIZE = (96, 96)

# ================== Flask ==================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ================== State ==================
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
is_processing = False
frame_count = 0
prediction_count = 0
last_prediction_time = 0
last_prediction_text = ""
PREDICTION_COOLDOWN = 3.0

# Emotion detection state
last_emotion_detection_time = time.time()
EMOTION_DETECTION_INTERVAL = 5.0  # 5 seconds
current_emotion = "neutral"
current_emotion_confidence = 0

# ================== MediaPipe ==================
mp_face_mesh = mp.solutions.face_mesh

def get_face_mesh():
    """Create a fresh MediaPipe FaceMesh instance"""
    return mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

print("✓ MediaPipe Face Mesh initialized")

# Lip landmarks indices
LIPS_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185
]

LIPS_INDICES = sorted(list(set(LIPS_INDICES)))


def extract_mouth_roi(frame, landmarks):
    """Extract ONLY the mouth region with tight bounding box"""
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
            print(f"[WARN] Only got {len(pts)} valid lip points")
            return None, None
        
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        lip_width = max(xs) - min(xs)
        lip_height = max(ys) - min(ys)
        
        padding_x = int(lip_width * 0.2)
        padding_y = int(lip_height * 0.2)
        
        x_min = max(0, min(xs) - padding_x)
        x_max = min(w, max(xs) + padding_x)
        y_min = max(0, min(ys) - padding_y)
        y_max = min(h, max(ys) + padding_y)
        
        if x_max <= x_min or y_max <= y_min:
            print(f"[WARN] Invalid ROI: ({x_min}, {y_min}) to ({x_max}, {y_max})")
            return None, None
        
        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            print(f"[WARN] Empty ROI")
            return None, None
        
        roi_h, roi_w = roi.shape[:2]
        
        scale = min(TARGET_SIZE[0] / roi_w, TARGET_SIZE[1] / roi_h)
        new_w = int(roi_w * scale)
        new_h = int(roi_h * scale)
        
        roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        canvas = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8)
        y_offset = (TARGET_SIZE[1] - new_h) // 2
        x_offset = (TARGET_SIZE[0] - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi_resized
        
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        print(f"[✓] Extracted mouth ROI: lips ({lip_width}x{lip_height}), bbox: ({x_min}, {y_min}, {x_max}, {y_max})")
        
        return gray, (x_min, y_min, x_max, y_max)
        
    except Exception as e:
        print(f"[ERROR] extract_mouth_roi: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def check_frame_diversity_improved(frame_buffer_deque):
    """Improved motion detection with lower thresholds"""
    if len(frame_buffer_deque) < FRAME_BUFFER_SIZE:
        return False
    
    frames = list(frame_buffer_deque)
    
    differences = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float)).mean()
        differences.append(diff)
    
    avg_diff = np.mean(differences)
    max_diff = np.max(differences)
    
    MIN_AVG_DIFF = 0.5
    MIN_MAX_DIFF = 1.0
    
    has_movement = avg_diff > MIN_AVG_DIFF and max_diff > MIN_MAX_DIFF
    
    print(f"[MOTION] Avg: {avg_diff:.2f} (min: {MIN_AVG_DIFF}), "
          f"Max: {max_diff:.2f} (min: {MIN_MAX_DIFF}), "
          f"Movement: {has_movement}")
    
    return has_movement

def calculate_frame_variance(frame_buffer_deque):
    """Calculate variance across frames to detect if mouth is moving"""
    if len(frame_buffer_deque) < FRAME_BUFFER_SIZE:
        return 0.0
    
    frames = np.array(list(frame_buffer_deque))
    variance = np.var(frames, axis=0).mean()
    
    print(f"[VARIANCE] Frame variance: {variance:.2f}")
    
    return variance


def process_frame(frame_data):
    """Process incoming frame - ENHANCED with lip detection and startup delay"""
    global frame_buffer, is_processing, frame_count, prediction_count
    global last_prediction_time, last_prediction_text
    global last_emotion_detection_time, current_emotion, current_emotion_confidence
    
    import time

    try:
        img_b64 = frame_data.split(',')[1] if ',' in frame_data else frame_data
        nparr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"type": "frame_result", "error": "Failed to decode frame"}

        frame_count += 1
        current_time = time.time()
        
        # Emotion detection every 5 seconds
        time_since_last_emotion = current_time - last_emotion_detection_time
        if time_since_last_emotion >= EMOTION_DETECTION_INTERVAL:
            print(f"\n{'='*50}")
            print(f"[EMOTION CHECK] Running emotion detection (interval: {time_since_last_emotion:.1f}s)")
            print(f"{'='*50}")
            
            emotion, confidence = detect_emotion_from_frame(frame)
            
            if emotion and emotion != 'neutral':
                current_emotion = emotion
                current_emotion_confidence = confidence
                print(f"[EMOTION UPDATE] ✓ Changed to: {emotion} ({confidence}%)")
            elif emotion == 'neutral':
                current_emotion = emotion
                current_emotion_confidence = confidence
                print(f"[EMOTION UPDATE] Neutral detected ({confidence}%)")
            else:
                print(f"[EMOTION UPDATE] ⚠ No emotion detected, keeping: {current_emotion}")
            
            last_emotion_detection_time = current_time
        
        face_mesh = get_face_mesh()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        face_mesh.close()

        payload = {
            "type": "frame_result",
            "face_detected": False,
            "mouth_roi": None,
            "prediction": None,
            "bbox": None,
            "debug": f"Frame {frame_count}",
            "emotion": current_emotion,
            "emotion_confidence": current_emotion_confidence
        }

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # ✅ NEW: Check if lips are open
            lips_open = are_lips_open(landmarks)
            
            if not lips_open:
                payload["debug"] = f"Frame {frame_count} | Lips closed - waiting for speech"
                payload["face_detected"] = True
                # Don't add to buffer if lips are closed
                return payload
            
            try:
                mouth_roi, bbox = extract_mouth_roi(frame, landmarks)
                
                if mouth_roi is not None:
                    frame_buffer.append(mouth_roi)

                    if RECORDING_MODE:
                        recorded_frames.append(mouth_roi.copy())
                        payload["debug"] = f"🔴 RECORDING '{RECORDING_WORD}': {len(recorded_frames)} frames"
                    
                    # ✅ NEW: Show startup grace period status
                    time_since_startup = time.time() - SESSION_START_TIME
                    if time_since_startup < STARTUP_GRACE_PERIOD:
                        remaining = STARTUP_GRACE_PERIOD - time_since_startup
                        payload["debug"] = f"⏳ Initializing ({remaining:.1f}s) | Buffer: {len(frame_buffer)}/{FRAME_BUFFER_SIZE}"
                    else:
                        payload["face_detected"] = True
                        payload["bbox"] = bbox
                        payload["debug"] = f"Frame {frame_count} | Buffer: {len(frame_buffer)}/{FRAME_BUFFER_SIZE} | Lips: OPEN"

                    thumb = cv2.resize(mouth_roi, (200, 200))
                    ok, buf = cv2.imencode(".jpg", thumb)
                    if ok:
                        payload["mouth_roi"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

                    time_since_last = current_time - last_prediction_time
                    
                    if len(frame_buffer) == FRAME_BUFFER_SIZE and not is_processing:
                        # ✅ NEW: Don't process during startup grace period
                        if time_since_startup < STARTUP_GRACE_PERIOD:
                            remaining = STARTUP_GRACE_PERIOD - time_since_startup
                            payload["debug"] = f"⏳ Warming up ({remaining:.1f}s remaining)"
                        elif time_since_last < PREDICTION_COOLDOWN:
                            payload["debug"] = f"Cooldown: {PREDICTION_COOLDOWN - time_since_last:.1f}s remaining"
                        else:
                            has_movement = check_frame_diversity_improved(frame_buffer)
                            variance = calculate_frame_variance(frame_buffer)
                            
                            if has_movement and variance > 2.0:
                                is_processing = True
                                try:
                                    print(f"\n{'='*60}")
                                    print(f"[🎬 RUNNING PREDICTION #{prediction_count + 1}]")
                                    print(f"{'='*60}")
                                    
                                    # This will now respect the startup grace period
                                    pred = predict_speech_avhubert(frame_buffer)
                                    
                                    if pred and len(pred.strip()) > 0:
                                        if pred != last_prediction_text or time_since_last > 5.0:
                                            prediction_count += 1
                                            last_prediction_time = current_time
                                            last_prediction_text = pred
                                            
                                            print(f"\n✓✓✓ PREDICTION #{prediction_count}: '{pred}' ✓✓✓\n")
                                            payload["prediction"] = pred
                                            
                                            frame_buffer.clear()
                                        else:
                                            print(f"\n⚠ Duplicate prediction ignored: '{pred}'\n")
                                            payload["debug"] = "Duplicate prediction - keep speaking"
                                    else:
                                        print(f"\n⚠ Prediction returned empty or None\n")
                                        payload["debug"] = "No speech detected - try speaking more clearly"
                                        
                                except Exception as e:
                                    print(f"[ERROR] Prediction failed: {e}")
                                    import traceback
                                    traceback.print_exc()
                                finally:
                                    is_processing = False
                            else:
                                payload["debug"] = f"Waiting for mouth movement (variance: {variance:.1f})"
                                # Clear some old frames if buffer is full but no movement
                                if len(frame_buffer) >= FRAME_BUFFER_SIZE:
                                    for _ in range(5):
                                        if frame_buffer:
                                            frame_buffer.popleft()
                else:
                    payload["debug"] = f"Frame {frame_count} | No mouth ROI extracted"
                    
            except Exception as e:
                print(f"[ERROR] Processing landmarks: {e}")
                payload["debug"] = f"Frame {frame_count} | Error: {str(e)}"
        else:
            payload["debug"] = f"Frame {frame_count} | No face detected"

        return payload
        
    except Exception as e:
        print(f"[ERROR] process_frame: {e}")
        import traceback
        traceback.print_exc()
        return {
            "type": "frame_result",
            "error": str(e),
            "face_detected": False,
            "mouth_roi": None,
            "prediction": None,
            "bbox": None,
            "emotion": current_emotion,
            "emotion_confidence": current_emotion_confidence
        }


def predict_speech_mock(frame_buffer_deque):
    """Mock prediction function that returns words from fixed vocabulary"""
    if len(frame_buffer_deque) < FRAME_BUFFER_SIZE:
        return None
    
    try:
        frames = list(frame_buffer_deque)
        
        print(f"[MOCK INFERENCE] Analyzing {len(frames)} frames...")
        
        frames_array = np.array(frames)
        
        mean_intensity = frames_array.mean()
        variance = frames_array.var()
        edge_density = np.abs(np.diff(frames_array, axis=0)).mean()
        
        seed_value = int((mean_intensity * 1000 + variance * 100 + edge_density * 10) % 1000)
        random.seed(seed_value)
        
        num_words = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        selected_words = random.sample(MOCK_VOCABULARY, num_words)
        prediction = " ".join(selected_words)
        
        print(f"[MOCK PREDICTION] Metrics - Mean: {mean_intensity:.2f}, Var: {variance:.2f}, Edge: {edge_density:.2f}")
        print(f"[MOCK PREDICTION] Seed: {seed_value}, Words: {num_words}")
        print(f"[MOCK PREDICTION] Text: '{prediction}'")
        
        return prediction
        
    except Exception as e:
        print(f"[ERROR] Mock prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_speech_avhubert(frame_buffer_deque):
    """Run prediction - delegates to template matching or real model"""
    if USE_TEMPLATE_MATCHING:
        print("[MODE] Using TEMPLATE MATCHING")
        return predict_using_templates(frame_buffer_deque)
    
    if USE_MOCK_PREDICTIONS:
        print("[MODE] Using MOCK predictions")
        return predict_speech_mock(frame_buffer_deque)
    
    # Real model prediction would go here
    print("[MODE] Real model not implemented in this snippet")
    return None


# ================== Routes ==================
@app.route("/")
def index():
    mode = "TEMPLATE MATCHING" if USE_TEMPLATE_MATCHING else ("MOCK MODE" if USE_MOCK_PREDICTIONS else "REAL MODEL")
    time_since_startup = time.time() - SESSION_START_TIME
    startup_status = "READY" if time_since_startup >= STARTUP_GRACE_PERIOD else f"WARMING UP ({STARTUP_GRACE_PERIOD - time_since_startup:.1f}s)"
    
    return jsonify({
        "status": "AV-HuBERT Lip Reading Server with ML Emotion Detection",
        "mode": mode,
        "startup_status": startup_status,
        "emotion_detection": "enabled" if EMOTION_DETECTION_AVAILABLE else "disabled",
        "frame_count": frame_count,
        "prediction_count": prediction_count,
        "templates_loaded": len(template_library) if USE_TEMPLATE_MATCHING else 0
    })


@app.route("/health")
def health():
    time_since_startup = time.time() - SESSION_START_TIME
    ready = time_since_startup >= STARTUP_GRACE_PERIOD
    
    return jsonify({
        "status": "healthy",
        "mode": "template_matching" if USE_TEMPLATE_MATCHING else ("mock" if USE_MOCK_PREDICTIONS else "real"),
        "ready": ready,
        "startup_grace_remaining": max(0, STARTUP_GRACE_PERIOD - time_since_startup),
        "emotion_detection": EMOTION_DETECTION_AVAILABLE,
        "frame_count": frame_count,
        "prediction_count": prediction_count,
        "buffer_size": len(frame_buffer),
        "templates_loaded": len(template_library)
    }), 200


@app.route("/process_frame", methods=["POST"])
def process_frame_http():
    global last_frame_time
    import time
    
    current_time = time.time()
    
    # Skip if too fast
    if current_time - last_frame_time < MIN_FRAME_INTERVAL:
        return jsonify({
            "type": "frame_result",
            "face_detected": False,
            "debug": "Rate limited"
        }), 200
    
    last_frame_time = current_time
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "No frame data provided"}), 400
        
        result = process_frame(data.get("frame"))
        
        if result is None:
            return jsonify({"error": "Processing failed", "type": "frame_result"}), 200
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[ERROR] Route handler: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "type": "frame_result",
            "face_detected": False
        }), 200

@app.route("/start_recording/<word>", methods=["POST"])
def start_recording_endpoint(word):
    """Start recording a template"""
    start_recording(word)
    return jsonify({
        "status": "recording", 
        "word": word,
        "message": f"Say '{word}' now!"
    })

@app.route("/stop_recording", methods=["POST"])
def stop_recording_endpoint():
    """Stop recording and save"""
    success, message = stop_recording()
    
    # Reload templates if successful
    if success:
        load_all_templates()
    
    return jsonify({
        "status": "success" if success else "error",
        "message": message
    })

@app.route("/recording_status", methods=["GET"])
def recording_status():
    """Check if currently recording"""
    global RECORDING_MODE, RECORDING_WORD, recorded_frames
    return jsonify({
        "recording": RECORDING_MODE,
        "word": RECORDING_WORD,
        "frames_collected": len(recorded_frames)
    })

# Add this route to your Flask app
@app.route("/send_report", methods=["POST"])
def send_report_endpoint():
    """
    Endpoint to send session report via email
    
    Expected JSON:
    {
        "email": "user@example.com",
        "report_data": {
            "predictions": [...],
            "emotions": [...],
            "duration": "5m 23s",
            "frame_count": 1234
        },
        "session_id": "optional-session-id"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        recipient_email = data.get('email')
        if not recipient_email:
            return jsonify({"error": "Email address required"}), 400
        
        report_data = data.get('report_data', {})
        session_id = data.get('session_id')
        
        # Send email
        success, message = send_report_email(recipient_email, report_data, session_id)
        
        if success:
            return jsonify({
                "status": "success",
                "message": message
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": message
            }), 500
            
    except Exception as e:
        print(f"[ERROR] send_report_endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ================== Startup ==================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎤 AV-HUBERT LIP READING SERVER WITH ENHANCEMENTS 🎤")
    print("="*70 + "\n")

    if EMOTION_DETECTION_AVAILABLE:
        print("✓ DeepFace Emotion Detection: ENABLED")
        print("  - Detection interval: Every 5 seconds")
        print("  - Emotions: happy, sad, angry, surprised, neutral, fearful, disgusted")
    else:
        print("⚠ DeepFace Emotion Detection: DISABLED")
    
    print()
    
    print("🆕 NEW FEATURES:")
    print(f"  ✓ Startup grace period: {STARTUP_GRACE_PERIOD} seconds")
    print("  ✓ Lip openness detection: Ignores closed lips")
    print("  ✓ Enhanced motion detection")
    print()
    
    if USE_TEMPLATE_MATCHING:
        print("🎯 RUNNING IN TEMPLATE MATCHING MODE")
        load_all_templates()
        print(f"📚 Loaded {len(template_library)} word templates")
    elif USE_MOCK_PREDICTIONS:
        print("⚠️  RUNNING IN MOCK MODE - Using fixed vocabulary")
        print(f"📚 Vocabulary: {len(MOCK_VOCABULARY)} words")
    else:
        print("🔄 RUNNING IN REAL MODEL MODE")
    
    print(f"\n🌐 Running on http://localhost:5056")
    print(f"📊 Frame buffer size: {FRAME_BUFFER_SIZE} frames")
    print(f"🎯 Target mouth ROI size: {TARGET_SIZE}")
    print(f"😊 Emotion Detection: {'ENABLED' if EMOTION_DETECTION_AVAILABLE else 'DISABLED'}")
    print(f"⏳ Startup Grace Period: {STARTUP_GRACE_PERIOD}s")
    print(f"👄 Lip Detection: ENABLED (ignores closed lips)")
    print("\n" + "="*70 + "\n")
    
    app.run(host="0.0.0.0", port=5056, debug=False, threaded=True)