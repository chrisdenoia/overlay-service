from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import io
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return 'Overlay service is live!'

@app.route('/process', methods=['POST'])
def process_pose():
    data = request.json

    # For now, simulate generating an image
    img = Image.new('RGB', (256, 256), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((50, 120), "Pose Processed", fill='black')

    # Convert image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({
        "status": "success",
        "overlay_base64": img_str
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)