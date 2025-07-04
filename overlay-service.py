# overlay-service.py

import os
import logging
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import mediapipe as mp
import io
import base64
import math

# 🔍 Debug Logging Setup
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"
def log(msg):
    if DEBUG_MODE:
        print(f"[DEBUG] {msg}")

app = Flask(__name__)

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_angle(a, b, c):
    """Calculates the angle between three points (in degrees)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return int(np.degrees(angle))

@app.route('/')
def home():
    return 'Overlay service is live!'

@app.route('/process', methods=['POST'])
def process_pose():
    try:
        data = request.json

        # Decode image
        image_bytes = base64.b64decode(data["image_base64"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(image)

        # MediaPipe processing
        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(np_image)

        draw = ImageDraw.Draw(image)
        angles = {}

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = image.height, image.width

            log(f"Image size: width={w}, height={h}")

            # Extract key points
            def get_coords(idx):
                pt = landmarks[idx]
                return (int(pt.x * w), int(pt.y * h))

            right_shoulder = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            right_elbow = get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)
            right_wrist = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)
            right_hip = get_coords(mp_pose.PoseLandmark.RIGHT_HIP)
            right_knee = get_coords(mp_pose.PoseLandmark.RIGHT_KNEE)

            log(f"right_shoulder: {right_shoulder}")
            log(f"right_elbow: {right_elbow}")
            log(f"right_wrist: {right_wrist}")
            log(f"right_hip: {right_hip}")
            log(f"right_knee: {right_knee}")

            # Angles
            angles["elbow"] = calculate_angle(right_shoulder, right_elbow, right_wrist)
            angles["shoulder"] = calculate_angle(right_hip, right_shoulder, right_elbow)
            angles["hip"] = calculate_angle(right_shoulder, right_hip, right_knee)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image=np_image,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Convert back to PIL for annotation
            image = Image.fromarray(np_image)
            draw = ImageDraw.Draw(image)

            # Font for angle display
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()

            # Draw angles
            draw.text(right_elbow, f"{angles['elbow']}°", fill="green", font=font)
            draw.text(right_shoulder, f"{angles['shoulder']}°", fill="blue", font=font)
            draw.text(right_hip, f"{angles['hip']}°", fill="red", font=font)

        # Encode output
        output = io.BytesIO()
        image.save(output, format="PNG")
        img_str = base64.b64encode(output.getvalue()).decode()

        return jsonify({
            "status": "success",
            "overlay_base64": img_str,
            "angles": angles
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)