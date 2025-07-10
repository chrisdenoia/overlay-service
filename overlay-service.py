# push
# overlay-service.py  –  v2025-07-09  (adds key-points JSON upload)
# ------------------------------------------------------------------
# What’s new?
#   1. Saves the 33 MediaPipe landmarks to Supabase Storage
#      as <user_id>/keypoints_<uuid>.json
#   2. Returns that public URL in the JSON response       ← keypoints_url
#   3. No other logic changed (same /process route, same overlay_base64)
# ------------------------------------------------------------------

import base64, io, os, uuid, json
import cv2
import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp
from PIL import Image
from supabase import create_client   # NEW

app = Flask(__name__)
mp_pose = mp.solutions.pose

# ---- Supabase client (service-role key gives write access) ----------
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)
BUCKET = "processed-data"            # same bucket you use for overlays

# ---------------------------------------------------------------------
def generate_pose_overlay(image_bytes):
    image_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_np)
        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected.")

        # ① Landmarks -> list[dict] for JSON
        lm_list = [
            {"x": lm.x, "y": lm.y, "z": lm.z or 0}
            for lm in results.pose_landmarks.landmark
        ]

        # ② Draw constellation overlay (unchanged)
        annotated_image = image_np.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    return annotated_image, lm_list
# ---------------------------------------------------------------------

@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json()
        image_base64 = data.get("image_base64")
        user_id      = data.get("user_id")  or "unknown-user"   # needs to be passed in
        if not image_base64:
            return jsonify(success=False, error="Missing image_base64"), 400

        image_bytes = base64.b64decode(image_base64)
        overlay_png, landmarks = generate_pose_overlay(image_bytes)

        # ---- Save keypoints JSON to Supabase Storage -----------------
        kp_path = f"{user_id}/keypoints_{uuid.uuid4()}.json"
        supabase.storage.from_(BUCKET).upload(
            kp_path,
            json.dumps(landmarks),
            { "contentType": "application/json", "upsert": True }
        )
        keypoints_url = supabase.storage.from_(BUCKET).get_public_url(kp_path)["public_url"]

        # ---- Encode overlay PNG back to base64 (unchanged) ----------
        _, buffer = cv2.imencode(".png", overlay_png)
        overlay_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify(
            success=True,
            overlay_base64=overlay_base64,
            keypoints_url=keypoints_url    # ← NEW FIELD
        )

    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=3000)