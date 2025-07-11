# overlay-service.py – v2025-07-10 (Supabase-py Response fix)
# ------------------------------------------------------------------
# Full file, no logs removed. Only change: handle new Response object
# returned by supabase.storage.upload().
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

# ---- Supabase client -------------------------------------------------
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)
BUCKET = "processed-data"

# ---------------------------------------------------------------------
# Helper: generate overlay + landmark list with full logging
# ---------------------------------------------------------------------

def generate_pose_overlay(image_bytes):
    print("[generate_pose_overlay] Decoding image → NumPy array …")
    rgb_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(rgb_np)
        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected.")

        landmarks = [
            {"x": lm.x, "y": lm.y, "z": lm.z or 0}
            for lm in results.pose_landmarks.landmark
        ]
        print(f"[generate_pose_overlay] Landmarks detected: {len(landmarks)}")

        annotated = rgb_np.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        print("[generate_pose_overlay] Overlay drawn.")

    return annotated, landmarks

# ---------------------------------------------------------------------
@app.route("/process", methods=["POST"])
def process():
    try:
        print("[process] Incoming request …")
        data = request.get_json()
        image_b64 = data.get("image_base64")
        user_id   = data.get("user_id")   or "unknown-user"
        upload_id = data.get("upload_id") or str(uuid.uuid4())
        print("[process] user_id:", user_id, "upload_id:", upload_id)

        if not image_b64:
            return jsonify(success=False, error="Missing image_base64"), 400

        img_bytes = base64.b64decode(image_b64)
        print("[process] Image decoded (bytes):", len(img_bytes))

        overlay_png, landmarks = generate_pose_overlay(img_bytes)

        # ---- Save keypoints JSON to Supabase Storage -----------------
        ts = int(time.time() * 1000)
        kp_path = f"{user_id}/keypoints_{upload_id}_{ts}.json"
        kp_bytes = json.dumps(landmarks, separators=(",", ":")).encode()
        print("[process] Uploading keypoints JSON →", kp_path)

        resp = supabase.storage.from_(BUCKET).upload(
            kp_path,
            kp_bytes,
            {"content-type": "application/json"},
        )
        if not getattr(resp, "is_success", False):
            print("[process] Storage upload error:", resp.status_code, resp.text)
            raise RuntimeError(f"Storage upload failed: {resp.status_code} – {resp.text}")

        keypoints_url = supabase.storage.from_(BUCKET).get_public_url(kp_path)[
            "publicUrl"
        ]
        print("[process] keypoints_url:", keypoints_url)

        # ---- Encode overlay PNG to base64 ---------------------------
        _, buf = cv2.imencode(".png", overlay_png)
        overlay_b64 = base64.b64encode(buf).decode()
        print("[process] overlay_base64 length:", len(overlay_b64))

        return jsonify(
            success=True,
            overlay_base64=overlay_b64,
            keypoints_url=keypoints_url,
        )

    except Exception as e:
        print("[process] ERROR:", e)
        return jsonify(success=False, error=str(e)), 500

# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=3000)
