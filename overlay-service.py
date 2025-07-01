from flask import Flask, request, jsonify, send_file
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
import io
import os
import base64

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

@app.route('/')
def index():
    return jsonify({"message": "Overlay service is running!"})

@app.route('/overlay', methods=['POST'])
def generate_overlay():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read and convert to RGB
    image = Image.open(file.stream).convert('RGB')
    img_array = np.array(image)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(img_array)

        if not results.pose_landmarks:
            return jsonify({'error': 'No pose landmarks detected'}), 400

        # Draw keypoints using PIL
        draw = ImageDraw.Draw(image)
        width, height = image.size

        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            draw.ellipse((x-3, y-3, x+3, y+3), fill='red')

    # Save result to buffer
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)