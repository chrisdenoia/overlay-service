#
import base64, io, os, uuid, json, time
import cv2
import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp
from PIL import Image
from supabase import create_client

app = Flask(__name__)

# 🚦 raise Flask’s hard cap so it will happily accept ~4 MB JSON bodies
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4 MB

mp_pose = mp.solutions.pose
mp_seg  = mp.solutions.selfie_segmentation  # NEW

# Supabase client (service-role key gives write access)
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)
BUCKET = "processed-data"

# ---------------------------------------------------------------------

def generate_pose_overlay(image_bytes):
    image_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    # 1️⃣ pose landmarks & constellation (existing)
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_np)
        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected.")
        # landmarks list
        lm_list = [
            {"x": lm.x, "y": lm.y, "z": lm.z or 0}
            for lm in results.pose_landmarks.landmark
        ]
        # draw skeleton
        annotated_image = image_np.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # 2️⃣ silhouette mask via SelfieSegmentation (NEW)
    with mp_seg.SelfieSegmentation(model_selection=1) as seg:
        seg_res = seg.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        mask = seg_res.segmentation_mask > 0.5
        silhouette = np.zeros_like(image_np)
        silhouette[mask] = (255, 255, 255)  # white body on black bg

    return annotated_image, lm_list, silhouette

# ---------------------------------------------------------------------

@app.route("/generate-pose-overlay", methods=["POST"])
def process():
    try:
        # 🔎——DEBUG: log the raw POST body *before* parsing ———————
        app.logger.warning(
            "RAW len=%s  first100=%s…",
            request.content_length,
            request.get_data()[:100]
        )
        # ————————————————————————————————————————————————

        data = request.get_json()
        image_base64 = data.get("image_base64")
        user_id      = data.get("user_id")  or "unknown-user"
        upload_id    = data.get("upload_id")
        if not image_base64 or not upload_id:
            return jsonify(success=False, error="Missing image or upload_id"), 400

        # decode
        image_bytes = base64.b64decode(image_base64)

        # generate constellation + landmarks + silhouette
        overlay_img, landmarks, silhouette_img = generate_pose_overlay(image_bytes)

        # ---- upload keypoints JSON (existing) ----
        kp_path = f"{user_id}/keypoints_{upload_id}_{int(time.time()*1000)}.json"
        kp_bytes = json.dumps(landmarks, separators=(",", ":")).encode()
        up = supabase.storage.from_(BUCKET).upload(
            kp_path, kp_bytes,
            {"contentType": "application/json", "upsert": True}
        )
        if up.get("error"):
            raise RuntimeError(f"Keypoints upload failed: {up['error']}")
        keypoints_url = supabase.storage.from_(BUCKET).get_public_url(kp_path)["publicUrl"]
        app.logger.info("Saved keypoints → %s", keypoints_url)

        # ---- upload constellation overlay (unchanged) ----
        _, const_buf = cv2.imencode(".png", cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        const_b64 = base64.b64encode(const_buf).decode("utf-8")

        # ---- upload silhouette PNG (NEW) ----
        _, sil_buf = cv2.imencode(".png", cv2.cvtColor(silhouette_img, cv2.COLOR_RGB2BGR))
        sil_path = f"{user_id}/overlay_silhouette_{upload_id}.png"
        sil_bytes = sil_buf.tobytes()
        up2 = supabase.storage.from_(BUCKET).upload(
            sil_path, sil_bytes,
            {"contentType": "image/png", "upsert": True}
        )
        if up2.get("error"):
            raise RuntimeError(f"Silhouette upload failed: {up2['error']}")
        silhouette_url = supabase.storage.from_(BUCKET).get_public_url(sil_path)["publicUrl"]
        app.logger.info("Saved silhouette → %s", silhouette_url)

        # ---- return everything to the edge-function caller ----
        return jsonify(
            success=True,
            overlay_base64=const_b64,
            keypoints_url=keypoints_url,
            overlay_silhouette_url=silhouette_url
        ), 200

    except Exception as e:
        app.logger.exception("Error in /process")
        return jsonify(success=False, error=str(e)), 500

# ---------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
