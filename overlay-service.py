# overlay-service.py  – v2025-07-10
# ------------------------------------------------------------------
# What’s new?
#   1. Generates a compact 33-landmark JSON file and uploads it to
#      Supabase Storage   →   keypoints_url public link.
#   2. Returns that URL in the /process JSON response.
#   3. Binds Flask on 0.0.0.0 so Railway can route traffic.
#   4. No original logs / validation / error handling removed.
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

# ---- Supabase client (service-role) -----------------------------------
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)
BUCKET = "processed-data"                    # same bucket as overlays

# ----------------------------------------------------------------------
def generate_pose_overlay(image_bytes):
    """
    • Detects pose landmarks
    • Returns:
        overlay_png  – NumPy array (BGR) with stick-figure overlay
        landmarks    – list[dict] of {x, y, z} MediaPipe points
    """
    image_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_np)
        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected.")

        # Landmarks to plain Python list for JSON
        landmarks = [
            {"x": lm.x, "y": lm.y, "z": lm.z or 0}
            for lm in results.pose_landmarks.landmark
        ]

        # Draw constellation overlay
        annotated = image_np.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    return annotated, landmarks
# ----------------------------------------------------------------------

@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json()
        image_base64 = data.get("image_base64")
        user_id   = data.get("user_id")   or "unknown-user"
        upload_id = data.get("upload_id") or str(uuid.uuid4())

        if not image_base64:
            return jsonify(success=False, error="Missing image_base64"), 400

        image_bytes = base64.b64decode(image_base64)
        overlay_png, landmarks = generate_pose_overlay(image_bytes)

        # ---- Save keypoints JSON to Supabase Storage ------------------
        timestamp   = int(time.time() * 1000)
        kp_path     = f"{user_id}/keypoints_{upload_id}_{timestamp}.json"
        kp_bytes    = json.dumps(landmarks, separators=(",", ":")).encode()

        up = supabase.storage.from_(BUCKET).upload(
            kp_path,
            kp_bytes,
            {"contentType": "application/json", "upsert": True}
        )
        if up.get("error"):
            raise RuntimeError(f"Storage upload error: {up['error']}")

        keypoints_url = supabase.storage.from_(BUCKET)\
                           .get_public_url(kp_path)["publicUrl"]
        print("Saved keypoints JSON →", keypoints_url)

        # ---- Encode overlay PNG back to base-64 ----------------------
        _, buffer = cv2.imencode(".png", overlay_png)
        overlay_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify(
            success=True,
            overlay_base64=overlay_b64,
            keypoints_url=keypoints_url
        )

    except Exception as e:
        # Standard error payload (unchanged)
        return jsonify(success=False, error=str(e)), 500

# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Bind on all interfaces so Railway can proxy traffic
    app.run(host="0.0.0.0", debug=True, port=3000)