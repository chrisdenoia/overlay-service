# -*- coding: utf-8 -*-
"""
overlay-service.py

Takes a base-64 image -> returns three artefacts:
- overlay_base64         - PNG of the pose constellation
- keypoints_url          - JSON with 33 landmarks
- overlay_silhouette_url - PNG RGBA silhouette (blue body / transparent bg)

All artefacts are saved in Supabase bucket 'processed-data'
inside a folder named after the upload_id.
"""

import base64
import io
import json
import os
import time

import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
import mediapipe as mp

# ---------- MediaPipe classic (constellation) ---------------------------------
MP_POSE = mp.solutions.pose

# ---------- MediaPipe Tasks (segmentation mask) --------------------------
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

_MODEL_PATH = "pose_landmarker_lite.task"  # make sure this file is in the repo

_BASE_OPTS = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
_POSE_OPTS = mp_vision.PoseLandmarkerOptions(
    base_options=_BASE_OPTS,
    output_segmentation_masks=True,
)
POSE_LANDMARKER = mp_vision.PoseLandmarker.create_from_options(_POSE_OPTS)

# ---------- Supabase client ----------------------------------------------
from supabase import create_client

SUPABASE = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)
BUCKET = "processed-data"

# ---------- Flask app -----------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB


# -------------------------------------------------------------------------
def generate_pose_overlay(image_bytes: bytes):
    """Return (constellation_rgb, landmark_list, silhouette_rgba)."""
    # Decode → RGB ndarray
    rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    # 1. Landmarks & constellation (classic solution)
    with MP_POSE.Pose(static_image_mode=True) as pose:
        res = pose.process(rgb)
        if not res.pose_landmarks:
            raise ValueError("No pose landmarks detected.")
        lm_list = [
            {"x": lm.x, "y": lm.y, "z": lm.z or 0.0}
            for lm in res.pose_landmarks.landmark
        ]
        constellation = rgb.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            constellation, res.pose_landmarks, MP_POSE.POSE_CONNECTIONS
        )

    # 2. Segmentation mask (MediaPipe Tasks)
    mp_img = mp.Image(                     # ← use mp.Image  ✅
        image_format=mp.ImageFormat.SRGB,  # ← and mp.ImageFormat ✅
        data=rgb,
    )
    det = POSE_LANDMARKER.detect(mp_img)
    seg_mask = det.segmentation_masks[0].numpy_view()  # float32 0-1

    # Binarise & clean with blur
    blurred = cv2.GaussianBlur(seg_mask, (5, 5), 0)
    mask = (blurred > 0.1).astype(np.uint8)
    k = max(3, int(0.02 * rgb.shape[0]))               # 2 % of height
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep largest blob
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_lbl > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (lbls == largest).astype(np.uint8)

    # Paint RGBA silhouette with transparent background
    h, w = mask.shape
    sil_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    sil_rgba[..., :3] = (66, 133, 244)            # Set RGB to blue
    sil_rgba[..., 3] = mask.astype(np.uint8) * 128  # semi-transparent blue

    # Composite silhouette over original image
    background = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.fromarray(sil_rgba, mode="RGBA")
    sil_rgba_composited = Image.alpha_composite(background, overlay)

    return constellation, lm_list, np.array(sil_rgba_composited)


# -------------------------------------------------------------------------
@app.route("/generate-pose-overlay", methods=["POST"])
def process():
    try:
        payload = request.get_json(force=True)
    except Exception as e:
        print(f"Error reading JSON: {str(e)}")
        return jsonify({"error": "Invalid or too large payload"}), 400
    img_b64 = payload.get("image_base64")
    upload_id = payload.get("upload_id")
    if not img_b64 or not upload_id:
        return jsonify(success=False, error="Missing image or upload_id"), 400

    try:
    img_bytes = base64.b64decode(img_b64, validate=True)
        except Exception as decode_err:
    print(f"Base64 decode error: {decode_err}")
        return jsonify(success=False, error="Invalid base64 image encoding"), 400

    try:
        constellation, landmarks, silhouette = generate_pose_overlay(img_bytes)
        # --- key-points JSON ------------------------------------------------
        kp_path = f"{upload_id}/keypoints_{int(time.time()*1000)}.json"
        kp_bytes = json.dumps(landmarks, separators=(",", ":")).encode()
        up = SUPABASE.storage.from_(BUCKET).upload(
            kp_path,
            kp_bytes,
            file_options={"content-type": "application/json", "x-upsert": "true"},
        )
        if up.status_code >= 400:
            raise RuntimeError(f"Key-points upload failed: {up.text!s}")
        kp_url  = SUPABASE.storage.from_(BUCKET).get_public_url(kp_path)

        # --- constellation overlay PNG (base-64 for caller) ----------------------
        _, buf = cv2.imencode(".png", cv2.cvtColor(constellation, cv2.COLOR_RGB2BGR))
        const_b64 = base64.b64encode(buf).decode()

        # --- silhouette PNG (RGBA→BGRA) ------------------------------------
        _, sil_buf = cv2.imencode(
            ".png", cv2.cvtColor(silhouette, cv2.COLOR_RGBA2BGRA)
        )
        sil_path = f"{upload_id}/overlay_silhouette.png"
        up2 = SUPABASE.storage.from_(BUCKET).upload(
            sil_path,
            sil_buf.tobytes(),
            file_options={"content-type": "image/png", "x-upsert": "true"},
        )
        if up2.status_code >= 400:
            raise RuntimeError(f"Silhouette upload failed: {up2.text!s}")
        sil_url = SUPABASE.storage.from_(BUCKET).get_public_url(sil_path)

        return jsonify(
            success=True,
            overlay_base64=const_b64,
            keypoints_url=kp_url,
            overlay_silhouette_url=sil_url,
        ), 200

    except Exception as exc:
        app.logger.exception("Error in /generate-pose-overlay")
        return jsonify(success=False, error=str(exc)), 500


# -------------------------------------------------------------------------
if __name__ == "__main__":  # local dev only
    app.run(host="0.0.0.0", port=3000, debug=True)