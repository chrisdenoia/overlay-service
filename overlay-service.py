# overlay-service.py  – v2025-07-10
# ------------------------------------------------------------------
# What’s new?
#   • Generates a compact 33‑landmark JSON and uploads it to Supabase
#     Storage → returns keypoints_url.
#   • Binds Flask on 0.0.0.0 for Railway.
#   • **upsert is now passed as the string "true"** to avoid bool‑to‑bytes
#     error.
# ------------------------------------------------------------------

import base64
import io
import os
import uuid
import json
import time

import cv2
import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp
from PIL import Image
from supabase import create_client

app = Flask(__name__)
mp_pose = mp.solutions.pose

# ---- Supabase client (service‑role) ---------------------------------
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)
BUCKET = "processed-data"  # same bucket used for overlays


# -------------------------------------------------------------------
# Helper: generate overlay + landmark list
# -------------------------------------------------------------------

def generate_pose_overlay(image_bytes):
    """Return (overlay_png_BGR, landmarks:list[dict])"""
    image_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_np)
        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected.")

        landmarks = [
            {"x": lm.x, "y": lm.y, "z": lm.z or 0}
            for lm in results.pose_landmarks.landmark
        ]

        annotated = image_np.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
    return annotated, landmarks


# -------------------------------------------------------------------
@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json()
        image_b64 = data.get("image_base64")
        user_id = data.get("user_id") or "unknown-user"
        upload_id = data.get("upload_id") or str(uuid.uuid4())

        if not image_b64:
            return jsonify(success=False, error="Missing image_base64"), 400

        img_bytes = base64.b64decode(image_b64)
        overlay_png, landmarks = generate_pose_overlay(img_bytes)

        # ---- Save keypoints JSON -----------------------------------
        ts = int(time.time() * 1000)
        kp_path = f"{user_id}/keypoints_{upload_id}_{ts}.json"
        kp_bytes = json.dumps(landmarks, separators=(",", ":")).encode()

        up = supabase.storage.from_(BUCKET).upload(
            kp_path,
            kp_bytes,
            {"content-type": "application/json", "upsert": "true"},  # <- FIXED
        )
        if up.get("error"):
            raise RuntimeError(f"Storage upload error: {up['error']}")

        keypoints_url = supabase.storage.from_(BUCKET).get_public_url(kp_path)[
            "publicUrl"
        ]
        print("Saved keypoints JSON →", keypoints_url)

        # ---- Overlay PNG to base64 ---------------------------------
        _, buf = cv2.imencode(".png", overlay_png)
        overlay_b64 = base64.b64encode(buf).decode("utf-8")

        return jsonify(
            success=True,
            overlay_base64=overlay_b64,
            keypoints_url=keypoints_url,
        )

    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=3000)
