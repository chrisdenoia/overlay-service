import os
import io
import base64
import math
import logging
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import mediapipe as mp

app = Flask(__name__)
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"

def log(msg):
    if DEBUG_MODE:
        print(f"[DEBUG] {msg}")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_angle(a, b, c):
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
        image_base64 = data.get("image_base64")
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(image)

        angles = {}
        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(np_image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = image.height, image.width

            def get_coords(idx):
                pt = landmarks[idx]
                return (int(pt.x * w), int(pt.y * h))

            # Get joints
            right_shoulder = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            right_elbow = get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)
            right_wrist = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)
            right_hip = get_coords(mp_pose.PoseLandmark.RIGHT_HIP)
            right_knee = get_coords(mp_pose.PoseLandmark.RIGHT_KNEE)

            # Angle calculations
            angles["elbow"] = calculate_angle(right_shoulder, right_elbow, right_wrist)
            angles["shoulder"] = calculate_angle(right_hip, right_shoulder, right_elbow)
            angles["hip"] = calculate_angle(right_shoulder, right_hip, right_knee)

            # Draw on np image first
            annotated_image = np_image.copy()
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Convert back to PIL
            image = Image.fromarray(annotated_image)
            draw = ImageDraw.Draw(image)

            # Draw angle text
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()

            draw.text(right_elbow, f"{angles['elbow']}°", fill="green", font=font)
            draw.text(right_shoulder, f"{angles['shoulder']}°", fill="blue", font=font)
            draw.text(right_hip, f"{angles['hip']}°", fill="red", font=font)
        else:
            log("⚠️ No pose detected")
            angles["error"] = "No pose landmarks found"

        # Encode overlay
        output = io.BytesIO()
        image.save(output, format="PNG")
        overlay_base64 = base64.b64encode(output.getvalue()).decode()

        return jsonify({
            "status": "success",
            "overlay_base64": overlay_base64,
            "angles": angles
        })
    except Exception as e:
        log(f"❌ Error: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)