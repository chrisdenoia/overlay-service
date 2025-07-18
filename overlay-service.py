# -*- coding: utf-8 -*-
"""
Complete Flask overlay-service.py with intelligent video processing

Features:
- Intelligent video processing with peak pose detection
- Single image analysis
- Overlay generation with existing keypoints
- Comprehensive error handling and logging
"""

import base64
import io
import json
import os
import time
import tempfile
from collections import deque

import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
import mediapipe as mp

# ---------- MediaPipe classic (constellation) ---------------------------------
MP_POSE = mp.solutions.pose

# ---------- MediaPipe Tasks (segmentation mask) --------------------------
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

_MODEL_PATH = "pose_landmarker_lite.task"  # make sure this file is in the repo

if os.path.exists(_MODEL_PATH):
    _BASE_OPTS = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
    _POSE_OPTS = mp_vision.PoseLandmarkerOptions(
        base_options=_BASE_OPTS,
        output_segmentation_masks=True,
    )
    POSE_LANDMARKER = mp_vision.PoseLandmarker.create_from_options(_POSE_OPTS)
else:
    print(f"Warning: {_MODEL_PATH} not found. Segmentation features will be disabled.")
    POSE_LANDMARKER = None

# ---------- Supabase client ----------------------------------------------
from supabase import create_client

SUPABASE = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)
BUCKET = "processed-data"

# ---------- Flask app -----------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB for videos

# ---------- Peak Pose Detection Functions --------------------------------

def calculate_pose_stability(keypoints_sequence, window_size=30):
    """Calculate pose stability over time using coefficient of variation."""
    if len(keypoints_sequence) < window_size:
        return 0.0
    
    # Focus on critical joints for stability analysis
    critical_joints = [11, 12, 13, 14, 23, 24, 25, 26]  # shoulders, elbows, hips, knees
    
    stability_scores = []
    
    for joint_idx in critical_joints:
        joint_positions = []
        for frame_keypoints in keypoints_sequence[-window_size:]:
            if joint_idx < len(frame_keypoints) and frame_keypoints[joint_idx]['visibility'] > 0.5:
                joint_positions.append([
                    frame_keypoints[joint_idx]['x'], 
                    frame_keypoints[joint_idx]['y']
                ])
        
        if len(joint_positions) < 10:  # Need sufficient data
            continue
            
        positions = np.array(joint_positions)
        
        # Calculate coefficient of variation for x and y
        mean_x, mean_y = np.mean(positions[:, 0]), np.mean(positions[:, 1])
        if mean_x > 0 and mean_y > 0:
            cv_x = np.std(positions[:, 0]) / mean_x
            cv_y = np.std(positions[:, 1]) / mean_y
            
            # Lower CV = higher stability
            joint_stability = 1.0 / (1.0 + cv_x + cv_y)
            stability_scores.append(joint_stability)
    
    return np.mean(stability_scores) if stability_scores else 0.0

def calculate_movement_velocity(keypoints_sequence):
    """Calculate overall movement velocity between consecutive frames."""
    if len(keypoints_sequence) < 2:
        return float('inf')
    
    prev_frame = keypoints_sequence[-2]
    curr_frame = keypoints_sequence[-1]
    
    total_movement = 0.0
    valid_joints = 0
    
    for i in range(min(len(prev_frame), len(curr_frame))):
        if (prev_frame[i]['visibility'] > 0.5 and 
            curr_frame[i]['visibility'] > 0.5):
            
            dx = curr_frame[i]['x'] - prev_frame[i]['x']
            dy = curr_frame[i]['y'] - prev_frame[i]['y']
            movement = np.sqrt(dx*dx + dy*dy)
            
            total_movement += movement
            valid_joints += 1
    
    return total_movement / valid_joints if valid_joints > 0 else float('inf')

def find_peak_pose_frame(video_path):
    """Find the frame with the most stable pose in a video."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    
    keypoints_sequence = deque(maxlen=60)  # Keep last 2 seconds at 30fps
    frame_analysis = []
    
    frame_idx = 0
    best_frame_idx = 0
    best_stability_score = 0.0
    best_frame_data = None
    
    # Skip early frames to allow person to get into position
    skip_frames = min(30, total_frames // 10)  # Skip first 10% or 30 frames, whichever is smaller
    
    with MP_POSE.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Skip early frames
            if frame_idx <= skip_frames:
                continue
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Convert landmarks to our format
                keypoints = [
                    {
                        "x": lm.x, 
                        "y": lm.y, 
                        "z": lm.z or 0.0,
                        "visibility": getattr(lm, 'visibility', 0.8)
                    }
                    for lm in results.pose_landmarks.landmark
                ]
                
                keypoints_sequence.append(keypoints)
                
                # Calculate metrics for this frame
                timestamp = frame_idx / fps
                confidence = np.mean([kp['visibility'] for kp in keypoints])
                
                # Calculate stability (need enough history)
                if len(keypoints_sequence) >= 30:
                    stability = calculate_pose_stability(keypoints_sequence)
                    velocity = calculate_movement_velocity(keypoints_sequence)
                    
                    # Combined score: high stability, low velocity, high confidence
                    velocity_score = 1.0 / (1.0 + velocity * 20)  # Penalize high movement
                    stability_score = (stability * 0.5 + velocity_score * 0.3 + confidence * 0.2)
                    
                    frame_analysis.append({
                        'frame_idx': frame_idx,
                        'timestamp': timestamp,
                        'stability': stability,
                        'velocity': velocity,
                        'confidence': confidence,
                        'score': stability_score
                    })
                    
                    # Check if this is the best frame so far
                    if stability_score > best_stability_score and confidence > 0.6:
                        best_stability_score = stability_score
                        best_frame_idx = frame_idx
                        best_frame_data = {
                            'keypoints': keypoints,
                            'frame': frame.copy(),
                            'timestamp': timestamp,
                            'stability': stability,
                            'confidence': confidence,
                            'score': stability_score
                        }
            
            # Progress logging
            if frame_idx % 60 == 0:
                progress = (frame_idx / total_frames) * 100
                best_time = best_frame_data['timestamp'] if best_frame_data else 0
                print(f"Progress: {progress:.1f}% - Current best at {best_time:.1f}s (score: {best_stability_score:.3f})")
    
    cap.release()
    
    if not best_frame_data:
        raise ValueError("No valid pose detected in video")
    
    print(f"Best frame found at {best_frame_data['timestamp']:.2f}s with stability score {best_stability_score:.3f}")
    
    return best_frame_data

def process_single_image(image_data):
    """Process a single image for pose detection."""
    # Decode image
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image")
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with MP_POSE.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    ) as pose:
        
        results = pose.process(rgb_image)
        
        if not results.pose_landmarks:
            raise ValueError("No pose detected in image")
        
        keypoints = [
            {
                "x": lm.x, 
                "y": lm.y, 
                "z": lm.z or 0.0,
                "visibility": getattr(lm, 'visibility', 0.8)
            }
            for lm in results.pose_landmarks.landmark
        ]
        
        confidence = np.mean([kp['visibility'] for kp in keypoints])
        
        return {
            'keypoints': keypoints,
            'frame': image,
            'timestamp': 0.0,
            'stability': 1.0,  # Single image = perfect stability
            'confidence': confidence,
            'score': confidence
        }

# ---------- Overlay Generation Functions --------------------------------

def create_landmark_from_keypoint(kp):
    """Convert keypoint dict to MediaPipe landmark for drawing."""
    class MockLandmark:
        def __init__(self, x, y, z, visibility=0.8):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = visibility
    return MockLandmark(kp['x'], kp['y'], kp.get('z', 0.0), kp.get('visibility', 0.8))

def create_pose_landmarks_from_keypoints(keypoints):
    """Convert keypoints list to MediaPipe pose landmarks for drawing."""
    class MockPoseLandmarks:
        def __init__(self, keypoints):
            self.landmark = [create_landmark_from_keypoint(kp) for kp in keypoints]
    return MockPoseLandmarks(keypoints)

def generate_pose_overlay_from_keypoints(image_bytes: bytes, keypoints: list):
    """Generate overlay using existing keypoints (no re-detection)."""
    # Decode → RGB ndarray
    rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    # 1. Draw constellation using existing keypoints
    constellation = rgb.copy()
    mock_landmarks = create_pose_landmarks_from_keypoints(keypoints)
    
    # Draw the pose constellation
    mp.solutions.drawing_utils.draw_landmarks(
        constellation, 
        mock_landmarks, 
        MP_POSE.POSE_CONNECTIONS,
        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
    )

    # 2. Generate segmentation mask for silhouette (if available)
    silhouette_rgba = None
    if POSE_LANDMARKER:
        try:
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )
            det = POSE_LANDMARKER.detect(mp_img)
            
            if det.segmentation_masks:
                seg_mask = det.segmentation_masks[0].numpy_view()  # float32 0-1

                # Binarise & clean with blur
                blurred = cv2.GaussianBlur(seg_mask, (5, 5), 0)
                mask = (blurred > 0.1).astype(np.uint8)
                k = max(3, int(0.02 * rgb.shape[0]))  # 2% of height
                kernel = np.ones((k, k), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                # Keep largest blob
                n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(mask)
                if n_lbl > 1:
                    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    mask = (lbls == largest).astype(np.uint8)

                # Paint RGBA silhouette with transparent background
                h, w = mask.shape
                sil_rgba = np.zeros((h, w, 4), dtype=np.uint8)
                sil_rgba[..., :3] = (66, 133, 244)  # Set RGB to blue
                sil_rgba[..., 3] = mask.astype(np.uint8) * 128  # semi-transparent blue

                # Composite silhouette over original image
                background = Image.fromarray(rgb).convert("RGBA")
                overlay = Image.fromarray(sil_rgba, mode="RGBA")
                silhouette_rgba = np.array(Image.alpha_composite(background, overlay))
        except Exception as e:
            print(f"Silhouette generation failed: {e}")

    return constellation, silhouette_rgba

def generate_pose_overlay_fallback(image_bytes: bytes):
    """Fallback: detect pose if keypoints not provided (legacy support)."""
    rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    with MP_POSE.Pose(static_image_mode=True) as pose:
        res = pose.process(rgb)
        if not res.pose_landmarks:
            raise ValueError("No pose landmarks detected.")
        
        keypoints = [
            {
                "x": lm.x, 
                "y": lm.y, 
                "z": lm.z or 0.0,
                "visibility": getattr(lm, 'visibility', 0.8)
            }
            for lm in res.pose_landmarks.landmark
        ]
        
        return generate_pose_overlay_from_keypoints(image_bytes, keypoints)

# ---------- Flask Routes ------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy", 
        "service": "pose-overlay",
        "mediapipe_pose": True,
        "mediapipe_tasks": POSE_LANDMARKER is not None,
        "timestamp": time.time()
    }), 200

@app.route("/process-pose-media", methods=["POST"])
def process_pose_media():
    """Process video or image to extract keypoints and optimal frame."""
    try:
        payload = request.get_json(force=True)
    except Exception as e:
        print(f"Error reading JSON: {str(e)}")
        return jsonify({"error": "Invalid or too large payload"}), 400
    
    upload_id = payload.get("upload_id")
    media_type = payload.get("media_type")  # 'video' or 'image'
    media_b64 = payload.get("media_base64")
    file_path = payload.get("file_path", "unknown")
    
    if not upload_id or not media_type or not media_b64:
        return jsonify(success=False, error="Missing required fields: upload_id, media_type, or media_base64"), 400
    
    start_time = time.time()
    
    try:
        # Decode media data
        media_data = base64.b64decode(media_b64, validate=True)
        
        print(f"Processing {media_type} for upload {upload_id}, size: {len(media_data)} bytes, file: {file_path}")
        
        # Process based on media type
        if media_type == 'video':
            # Create temporary file for video processing
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_video_path = temp_file.name
                temp_file.write(media_data)
            
            try:
                # Find peak pose frame in video
                result_data = find_peak_pose_frame(temp_video_path)
                processing_method = "video_peak_detection"
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        
        elif media_type == 'image':
            # Process single image
            result_data = process_single_image(media_data)
            processing_method = "single_image_analysis"
        
        else:
            return jsonify(success=False, error=f"Unsupported media type: {media_type}"), 400
        
        # Convert frame to base64
        _, buffer = cv2.imencode('.png', result_data['frame'])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        processing_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            "success": True,
            "keypoints": result_data['keypoints'],
            "extracted_frame_base64": frame_base64,
            "peak_frame_time": result_data['timestamp'],
            "stability_score": result_data['stability'],
            "confidence_score": result_data['confidence'],
            "processing_method": processing_method,
            "keypoints_count": len(result_data['keypoints']),
            "processing_time_seconds": processing_time
        }
        
        print(f"Successfully processed {media_type}: {len(result_data['keypoints'])} keypoints, "
              f"confidence: {result_data['confidence']:.3f}, "
              f"stability: {result_data['stability']:.3f}, "
              f"time: {processing_time:.2f}s")
        
        return jsonify(response_data), 200
        
    except Exception as exc:
        processing_time = time.time() - start_time
        app.logger.exception("Error in /process-pose-media")
        print(f"Processing failed after {processing_time:.2f}s: {str(exc)}")
        return jsonify(
            success=False, 
            error=str(exc),
            processing_method="failed",
            processing_time_seconds=processing_time
        ), 500

@app.route("/generate-pose-overlay", methods=["POST"])
def generate_pose_overlay():
    """Generate pose overlay with existing keypoints or detect new ones."""
    try:
        payload = request.get_json(force=True)
    except Exception as e:
        print(f"Error reading JSON: {str(e)}")
        return jsonify({"error": "Invalid or too large payload"}), 400

    img_b64 = payload.get("image_base64")
    upload_id = payload.get("upload_id")
    keypoints = payload.get("keypoints")  # Optional: use existing keypoints
    
    if not img_b64 or not upload_id:
        return jsonify(success=False, error="Missing image_base64 or upload_id"), 400

    start_time = time.time()

    try:
        img_bytes = base64.b64decode(img_b64, validate=True)
    except Exception as decode_err:
        print(f"Base64 decode error: {decode_err}")
        return jsonify(success=False, error="Invalid base64 image encoding"), 400

    try:
        # Use existing keypoints if provided, otherwise detect (fallback)
        if keypoints and len(keypoints) == 33:
            print("Using provided keypoints from pose detection")
            constellation, silhouette = generate_pose_overlay_from_keypoints(img_bytes, keypoints)
            landmarks_detected = True
            pose_detected = True
        else:
            print("No keypoints provided, falling back to pose detection")
            constellation, silhouette = generate_pose_overlay_fallback(img_bytes)
            landmarks_detected = True
            pose_detected = True

        # --- constellation overlay PNG (base-64 for caller) ------------------
        _, buf = cv2.imencode(".png", cv2.cvtColor(constellation, cv2.COLOR_RGB2BGR))
        const_b64 = base64.b64encode(buf).decode()

        # --- silhouette handling -------------------------------------
        silhouette_url = None
        silhouette_b64 = None
        
        if silhouette is not None:
            try:
                # Upload silhouette to storage
                _, sil_buf = cv2.imencode(".png", cv2.cvtColor(silhouette, cv2.COLOR_RGBA2BGRA))
                sil_path = f"{upload_id}/overlay_silhouette.png"
                
                up2 = SUPABASE.storage.from_(BUCKET).upload(
                    sil_path,
                    sil_buf.tobytes(),
                    file_options={"content-type": "image/png", "x-upsert": "true"},
                )
                
                if up2.status_code < 400:
                    silhouette_url = SUPABASE.storage.from_(BUCKET).get_public_url(sil_path)
                    print(f"Silhouette uploaded successfully to: {silhouette_url}")
                else:
                    print(f"Silhouette upload failed: {up2.text}")
                    # Fallback to base64
                    silhouette_b64 = base64.b64encode(sil_buf).decode()
                    
            except Exception as sil_error:
                print(f"Silhouette processing error: {sil_error}")
                # Continue without silhouette

        processing_time = time.time() - start_time

        response_data = {
            "success": True,
            "processed_successfully": True,
            "pose_detected": pose_detected,
            "landmarks_found": 33 if landmarks_detected else 0,
            "overlay_base64": const_b64,
            "processing_time_seconds": processing_time
        }
        
        # Add silhouette data if available
        if silhouette_url:
            response_data["overlay_silhouette_url"] = silhouette_url
        elif silhouette_b64:
            response_data["silhouette_base64"] = silhouette_b64

        print(f"Overlay generation completed in {processing_time:.2f}s")
        
        return jsonify(response_data), 200

    except Exception as exc:
        processing_time = time.time() - start_time
        app.logger.exception("Error in /generate-pose-overlay")
        print(f"Overlay generation failed after {processing_time:.2f}s: {str(exc)}")
        return jsonify(
            success=False, 
            processed_successfully=False,
            error_message=str(exc),
            processing_time_seconds=processing_time
        ), 500

@app.route("/analyze-pose", methods=["POST"])
def analyze_pose():
    """Extract keypoints only from a single image (lightweight endpoint)."""
    try:
        payload = request.get_json(force=True)
    except Exception as e:
        print(f"Error reading JSON: {str(e)}")
        return jsonify({"error": "Invalid or too large payload"}), 400
    
    img_b64 = payload.get("image_base64")
    
    if not img_b64:
        return jsonify(success=False, error="Missing image_base64"), 400

    start_time = time.time()

    try:
        img_bytes = base64.b64decode(img_b64, validate=True)
        
        # Process single image
        result_data = process_single_image(img_bytes)
        
        processing_time = time.time() - start_time
        
        return jsonify(
            success=True,
            keypoints=result_data['keypoints'],
            landmarks_count=len(result_data['keypoints']),
            confidence_score=result_data['confidence'],
            processing_time_seconds=processing_time
        ), 200

    except Exception as exc:
        processing_time = time.time() - start_time
        app.logger.exception("Error in /analyze-pose")
        print(f"Pose analysis failed after {processing_time:.2f}s: {str(exc)}")
        return jsonify(
            success=False, 
            error=str(exc),
            processing_time_seconds=processing_time
        ), 500

# ---------- Error Handlers ----------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": "File too large. Maximum size allowed is 200MB.",
        "error_code": 413
    }), 413

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "success": False,
        "error": "Bad request. Please check your input data.",
        "error_code": 400
    }), 400

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error. Please try again later.",
        "error_code": 500
    }), 500

# ---------- Main ---------------------------------------------------------

if __name__ == "__main__":
    # Development server only
    print("🚀 Starting Flask pose processing service...")
    print(f"MediaPipe Pose: ✅")
    print(f"MediaPipe Tasks (Segmentation): {'✅' if POSE_LANDMARKER else '❌'}")
    print(f"Supabase Connection: ✅")
    print("🎯 Ready for intelligent video and image processing!")
    
    app.run(host="0.0.0.0", port=3000, debug=True)