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
def _extract_url(resp):
    """
    get_public_url() may return either a dict or a plain string.
    Always return the URL string.
    """
    if isinstance(resp, str):
        return resp
    return (
        resp.get("publicUrl")
        or resp.get("data", {}).get("publicUrl")
        or "UNKNOWN_URL"
    )

# ---------------------------------------------------------------------

def generate_pose_overlay(image_bytes):
    image_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    # 1️⃣ pose landmarks & constellation
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_np)
        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected.")

        lm_list = [
            {"x": lm.x, "y": lm.y, "z": lm.z or 0}
            for lm in results.pose_landmarks.landmark
        ]

        annotated_image = image_np.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # 2️⃣ silhouette mask via SelfieSegmentation  ──────────────────────────
    with mp_seg.SelfieSegmentation(model_selection=1) as seg:
        seg_res  = seg.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        raw_mask = seg_res.segmentation_mask                     # float 0-1

        # ▸ threshold + hole fill
        mask_bin = (raw_mask > 0.10).astype(np.uint8)
        k        = max(3, int(0.01 * image_np.shape[0]))         # 1 % of height
        kernel   = np.ones((k, k), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

        # ▸ build **RGBA** overlay – MediaPipe blue on transparent bg
        h, w, _  = image_np.shape
        silhouette = np.zeros((h, w, 4), dtype=np.uint8)         # RGBA
        silhouette[mask_bin.astype(bool)] = (66, 133, 244, 255)  # (#4285F4, α=255)

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
        app.logger.warning("DEBUG types – base64:%s  uid:%s  upload:%s",
                   type(data.get("image_base64")).__name__,
                   type(data.get("user_id")).__name__,
                   type(data.get("upload_id")).__name__)
        image_base64 = data.get("image_base64")
        user_id      = data.get("user_id")  or "unknown-user"
        upload_id    = data.get("upload_id")
        if not image_base64 or not upload_id:
            return jsonify(success=False, error="Missing image or upload_id"), 400

        # decode
        image_bytes = base64.b64decode(image_base64)

        # generate constellation + landmarks + silhouette
        overlay_img, landmarks, silhouette_img = generate_pose_overlay(image_bytes)

        # ---- upload KEY-POINTS JSON ----------------------------------------
        kp_path  = f"{user_id}/keypoints_{upload_id}_{int(time.time()*1000)}.json"
        kp_bytes = json.dumps(landmarks, separators=(",", ":")).encode()

        up = supabase.storage.from_(BUCKET).upload(
            kp_path,
            kp_bytes,
            file_options={                 # one dict of headers ✅
                "content-type": "application/json",
                "x-upsert":     "true"
            }
        )
        if up.status_code >= 400:                       # ← no .get(), use HTTP status
            raise RuntimeError(f"Upload failed – {up.status_code}: {up.text}")

        keypoints_url = _extract_url(
            supabase.storage.from_(BUCKET).get_public_url(kp_path)
        )
        app.logger.info("Saved key-points → %s", keypoints_url)

        # ---- encode CONSTELLATION overlay (base64 for the caller) ----------
        _, const_buf = cv2.imencode(".png",
                                    cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        const_b64 = base64.b64encode(const_buf).decode("utf-8")

        # ---- upload silhouette PNG ----------------------------------------
        _, sil_buf = cv2.imencode(".png", silhouette_img)      # keep alpha
        sil_path   = f"{user_id}/overlay_silhouette_{upload_id}.png"

        up2 = supabase.storage.from_(BUCKET).upload(
            sil_path,
            sil_buf.tobytes(),
            # put every option in this ONE dict – note the header name
            file_options={
                "content-type": "image/png",
                "x-upsert":     "true"        # <-- header flag, **string**
            }
        )
   if up2.status_code >= 400:             # <-- Response-object, not dict
    raise RuntimeError(f"Silhouette upload failed: {up2.text}")

        silhouette_url = supabase.storage.from_(BUCKET)\
                                         .get_public_url(sil_path)["publicUrl"]
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
