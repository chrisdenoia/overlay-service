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

    # Run MediaPipe Pose inside context manager
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(np_image)

    # Draw pose landmarks if available
    draw = ImageDraw.Draw(image)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            x = int(lm.x * image.width)
            y = int(lm.y * image.height)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill='blue')

    # Convert processed image to base64
    output = io.BytesIO()
    image.save(output, format="PNG")
    img_str = base64.b64encode(output.getvalue()).decode()

    return jsonify({
        "status": "success",
        "overlay_base64": img_str
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)