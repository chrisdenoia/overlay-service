import os
import logging
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import mediapipe as mp
import io
import base64
import math

DEBUG_MODE = True
def log(msg):
    if DEBUG_MODE:
        print(f"[DEBUG] {msg}")

app = Flask(__name__)
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
        log("✅ Received request to /process")
        log(f"📦 image_base64 length: {len(data['image_base64'])}")

        image_bytes = base64.b64decode(data["image_base64"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(image)
        log(f"🖼️ Image converted, size={image.size}")

        angles = {}
        pose_detected = False
        landmark_count = 0

        with mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(np_image)

        if results.pose_landmarks:
            pose_detected = True
            landmarks = results.pose_landmarks.landmark
            landmark_count = len(landmarks)
            h, w = image.height, image.width

            def get_coords(idx):
                pt = landmarks[idx]
                return (int(pt.x * w), int(pt.y * h))

            right_shoulder = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            right_elbow = get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)
            right_wrist = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)
            right_hip = get_coords(mp_pose.PoseLandmark.RIGHT_HIP)
            right_knee = get_coords(mp_pose.PoseLandmark.RIGHT_KNEE)

            angles["elbow"] = calculate_angle(right_shoulder, right_elbow, right_wrist)
            angles["shoulder"] = calculate_angle(right_hip, right_shoulder, right_elbow)
            angles["hip"] = calculate_angle(right_shoulder, right_hip, right_knee)

            mp_drawing.draw_landmarks(
                image=np_image,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            image = Image.fromarray(np_image)
            draw = ImageDraw.Draw(image)

            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()

            draw.text(right_elbow, f"{angles['elbow']}°", fill="green", font=font)
            draw.text(right_shoulder, f"{angles['shoulder']}°", fill="blue", font=font)
            draw.text(right_hip, f"{angles['hip']}°", fill="red", font=font)

        else:
            log("⚠️ No pose detected! MediaPipe returned no landmarks.")

        log(f"📊 Pose detected: {pose_detected}")
        log(f"📈 Landmark count: {landmark_count}")
        log(f"📐 Angles calculated: {angles}")

        output = io.BytesIO()
        image.save(output, format="PNG")
        img_str = base64.b64encode(output.getvalue()).decode()

        return jsonify({
            "processed_successfully": pose_detected,
            "pose_detected": pose_detected,
            "landmarks_found": landmark_count,
            "angles": angles,
            "overlay_base64": img_str
        })
    except Exception as e:
        log(f"❌ Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

# ✅ Replaces: app.run(host='0.0.0.0', port=3000)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run(host='0.0.0.0', port=port)