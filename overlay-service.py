"""
overlay-service.py
──────────────────
Given a base-64 image, this service returns three artefacts:

  • overlay_base64        – PNG of the pose skeleton (constellation)
  • keypoints_url         – JSON file with 33 MediaPipe landmarks
  • overlay_silhouette_url– PNG RGBA silhouette (blue body / transparent bg)

All artefacts are stored in Supabase Storage bucket :processed-data
under a folder named after the upload_id.
"""

import base64, io, json, os, time
import cv2, numpy as np
from PIL import Image
from flask import Flask, jsonify, request
import mediapipe as mp

# ───────── MediaPipe classic (for skeleton) ───────────────────────────
MP_POSE = mp.solutions.pose

# ───────── MediaPipe Tasks (for segmentation mask) ────────────────────
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

_MODEL_PATH = "pose_landmarker_lite.task"          # bundle this in the repo

_BASE_OPTS  = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
_POSE_OPTS  = mp_vision.PoseLandmarkerOptions(
    base_options=_BASE_OPTS,
    output_segmentation_masks=True,
)
POSE_LMKR   = mp_vision.PoseLandmarker.create_from_options(_POSE_OPTS)

# ───────── Supabase client ────────────────────────────────────────────
from supabase import create_client

SUPABASE = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)
BUCKET   = "processed-data"

# ───────── Flask app ──────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024        # 4 MB JSON bodies


# =====================================================================
#  Core helper
# =====================================================================
def generate_pose_overlay(image_bytes: bytes):
    """Return (skeleton_rgb, landmark_list, silhouette_rgba)"""
    # decode → RGB ndarray
    rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    # 1️⃣ Landmarks & constellation (classic solution) -----------------
    with MP_POSE.Pose(static_image_mode=True) as pose:
        res = pose.process(rgb)
        if not res.pose_landmarks:
            raise ValueError("No pose landmarks detected.")
        lm_list = [
            {"x": lm.x, "y": lm.y, "z": lm.z or 0.0}
            for lm in res.pose_landmarks.landmark
        ]
        skeleton = rgb.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            skeleton, res.pose_landmarks, MP_POSE.POSE_CONNECTIONS
        )

    # 2️⃣ Segmentation mask (Tasks Pose Landmarker) --------------------
    mp_img   = mp_vision.Image(image_format=mp_vision.ImageFormat.SRGB, data=rgb)
    det      = POSE_LMKR.detect(mp_img)
    seg_mask = det.segmentation_masks[0].numpy_view()      # float32 [0…1]

    # Binarise & clean
    mask = (seg_mask > 0.05).astype(np.uint8)              # low threshold
    k    = max(3, int(0.02 * rgb.shape[0]))                # 2 % of height
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    # keep largest blob
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_lbl > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask    = (lbls == largest).astype(np.uint8)

    # paint RGBA silhouette – MediaPipe blue
    h, w, _  = rgb.shape
    sil_rgba = np.zeros((h, w, 4), np.uint8)
    sil_rgba[mask.astype(bool)] = (66, 133, 244, 255)      # #4285F4

    return skeleton, lm_list, sil_rgba


# =====================================================================
#  HTTP endpoint
# =====================================================================
@app.route("/generate-pose-overlay", methods=["POST"])
def process():
    try:
        payload = request.get_json(force=True)
        img_b64   = payload.get("image_base64")
        upload_id = payload.get("upload_id")
        if not img_b64 or not upload_id:
            return jsonify(success=False, error="Missing image or upload_id"), 400

        # decode JPEG/PNG → bytes
        img_bytes = base64.b64decode(img_b64)

        # build artefacts
        overlay_img, landmarks, silhouette = generate_pose_overlay(img_bytes)

        # ── key-points JSON --------------------------------------------------
        kp_path  = f"{upload_id}/keypoints_{int(time.time()*1000)}.json"
        kp_bytes = json.dumps(landmarks, separators=(",", ":")).encode()
        up = SUPABASE.storage.from_(BUCKET).upload(
            kp_path,
            kp_bytes,
            file_options={"content-type": "application/json", "x-upsert": "true"},
        )
        if up.status_code >= 400:
            raise RuntimeError(f"Key-points upload failed: {up.text!s}")
        kp_url = SUPABASE.storage.from_(BUCKET).get_public_url(kp_path)["publicUrl"]

        # ── constellation PNG (base-64 back to caller) ----------------------
        _, buf = cv2.imencode(".png", cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        const_b64 = base64.b64encode(buf).decode()

        # ── silhouette PNG (RGBA→BGRA) -------------------------------------
        _, sil_buf = cv2.imencode(".png", cv2.cvtColor(silhouette, cv2.COLOR_RGBA2BGRA))
        sil_path  = f"{upload_id}/overlay_silhouette.png"
        up2 = SUPABASE.storage.from_(BUCKET).upload(
            sil_path,
            sil_buf.tobytes(),
            file_options={"content-type": "image/png", "x-upsert": "true"},
        )
        if up2.status_code >= 400:
            raise RuntimeError(f"Silhouette upload failed: {up2.text!s}")
        sil_url = SUPABASE.storage.from_(BUCKET).get_public_url(sil_path)["publicUrl"]

        # ────────────────────────────────────────────────────────────────────
        return jsonify(
            success=True,
            overlay_base64=const_b64,
            keypoints_url=kp_url,
            overlay_silhouette_url=sil_url,
        ), 200

    except Exception as exc:
        app.logger.exception("Error in /generate-pose-overlay")
        return jsonify(success=False, error=str(exc)), 500


# =====================================================================
if __name__ == "__main__":               # local dev / debugging only
    app.run(host="0.0.0.0", port=3000, debug=True)