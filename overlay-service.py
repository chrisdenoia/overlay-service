# overlay-service.py
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import numpy as np
import mediapipe as mp
import io
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return 'Overlay service is live!'

@app.route('/process', methods=['POST'])
def process_pose():
    data = request.json

    # Decode image from base64
    image_bytes = base64.b64decode(data["image_base64"])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(image)

    # Run MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    results = pose.process(np_image)

    # Draw keypoints and skeleton
    draw = ImageDraw.Draw(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        width, height = image.size

        # Draw joints (keypoints)
        for lm in landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='blue')

        # Draw lines between key landmarks
        connections = mp_pose.POSE_CONNECTIONS
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            x0, y0 = int(start.x * width), int(start.y * height)
            x1, y1 = int(end.x * width), int(end.y * height)
            draw.line((x0, y0, x1, y1), fill='lime', width=2)

    # Convert to base64 and return
    output = io.BytesIO()
    image.save(output, format="PNG")
    img_str = base64.b64encode(output.getvalue()).decode()

    return jsonify({
        "status": "success",
        "overlay_base64": img_str
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
