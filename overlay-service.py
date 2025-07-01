from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import mediapipe as mp
import uuid
import os

app = Flask(__name__)

OUTPUT_DIR = 'output_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

@app.route('/overlay', methods=['POST'])
def generate_overlay():
    try:
        data = request.get_json()
        image_data = data.get('imageBase64')
        upload_id = data.get('upload_id')

        if not image_data or not upload_id:
            return jsonify({'error': 'Missing imageBase64 or upload_id'}), 400

        image_bytes = base64.b64decode(image_data)
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return jsonify({'error': 'Pose not detected'}), 422

        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        output_path = os.path.join(OUTPUT_DIR, f"{upload_id}_overlay.png")
        cv2.imwrite(output_path, annotated_image)

        return jsonify({
            'status': 'success',
            'overlay_image_path': output_path
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)