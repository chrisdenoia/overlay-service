# overlay-service.py – v2025-07-11 (adds /generate-pose-overlay route)
# ------------------------------------------------------------------
# Endpoints
#  • POST /process                 – existing image-base64 workflow (unchanged)
#  • POST /generate-pose-overlay   – NEW. Accepts a video_url + callback,
#                                    grabs the first frame, runs MediaPipe,
#                                    saves overlay PNG + keypoints JSON,
#                                    optionally POSTs results to callback.
# No logging or validation removed.
# ------------------------------------------------------------------

import base64, io, os, uuid, json, time, tempfile, urllib.request
import cv2, numpy as np, requests
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
# 1) /process – accepts base-64 image (unchanged except for last fix)
# ---------------------------------------------------------------------

@app.route("/process", methods=["POST"])
def process():
    try:
        print("[process] Incoming /process request …")
        data = request.get_json()
        image_b64 = data.get("image_base64")
        user_id   = data.get("user_id")   or "unknown-user"
        upload_id = data.get("upload_id") or str(uuid.uuid4())
        print("[process] user_id:", user_id, "upload_id:", upload_id)

        if not image_b64:
            return jsonify(success=False, error="Missing image_base64"), 400

        img_bytes = base64.b64decode(image_b64)
        overlay_png, landmarks = generate_pose_overlay(img_bytes)

        keypoints_url = save_keypoints_and_get_url(landmarks, user_id, upload_id)
        overlay_b64   = encode_png_to_base64(overlay_png)

        return jsonify(success=True, overlay_base64=overlay_b64, keypoints_url=keypoints_url)

    except Exception as e:
        print("[process] ERROR:", e)
        return jsonify(success=False, error=str(e)), 500

# ---------------------------------------------------------------------
# 2) /generate-pose-overlay – NEW video workflow
# ---------------------------------------------------------------------

@app.route("/generate-pose-overlay", methods=["POST"])
def generate_pose_overlay_route():
    try:
        print("[gen-overlay] Incoming /generate-pose-overlay request …")
        data = request.get_json()
        video_url   = data.get("video_url")
        upload_id   = data.get("upload_id") or str(uuid.uuid4())
        user_id     = data.get("user_id")   or "unknown-user"
        callback_url = data.get("callback_url")
        print("[gen-overlay] video_url:", video_url)

        if not video_url:
            return jsonify(success=False, error="Missing video_url"), 400

        # 1. download the video to a temp file
        tmp_path = tempfile.mktemp(suffix=".mp4")
        print("[gen-overlay] Downloading video →", tmp_path)
        urllib.request.urlretrieve(video_url, tmp_path)

        # 2. capture first frame
        cap = cv2.VideoCapture(tmp_path)
        ok, frame = cap.read()
        cap.release()
        os.remove(tmp_path)
        if not ok:
            return jsonify(success=False, error="Unable to read video"), 400
        print("[gen-overlay] First frame captured – running MediaPipe …")

        # convert frame (BGR) to bytes as JPEG
        _, jpg_buf = cv2.imencode(".jpg", frame)
        overlay_png, landmarks = generate_pose_overlay(jpg_buf.tobytes())

        keypoints_url = save_keypoints_and_get_url(landmarks, user_id, upload_id)
        overlay_b64   = encode_png_to_base64(overlay_png)

        # 3. optional callback
        if callback_url:
            print("[gen-overlay] POSTing results → callback_url")
            try:
                requests.post(callback_url, json={
                    "upload_id": upload_id,
                    "overlay_base64": overlay_b64,
                    "keypoints_url": keypoints_url
                }, timeout=10)
            except Exception as cb_err:
                print("[gen-overlay] callback failed:", cb_err)

        return jsonify(success=True, overlay_base64=overlay_b64, keypoints_url=keypoints_url)

    except Exception as e:
        print("[gen-overlay] ERROR:", e)
        return jsonify(success=False, error=str(e)), 500

# ---------------------------------------------------------------------
# Helper: save keypoints JSON + return public URL
# ---------------------------------------------------------------------

def save_keypoints_and_get_url(landmarks, user_id, upload_id):
    ts = int(time.time() * 1000)
    kp_path = f"{user_id}/keypoints_{upload_id}_{ts}.json"
    kp_bytes = json.dumps(landmarks, separators=(",", ":")).encode()
    print("[helper] Uploading keypoints JSON →", kp_path)

    resp = supabase.storage.from_(BUCKET).upload(
        kp_path,
        kp_bytes,
        {"content-type": "application/json"},
    )
    if not getattr(resp, "is_success", False):
        raise RuntimeError(f"Storage upload failed: {resp.status_code} – {resp.text}")

    raw_url = supabase.storage.from_(BUCKET).get_public_url(kp_path)
    if isinstance(raw_url, str):
        return raw_url
    return raw_url.get("publicUrl") or raw_url.get("public_url")

# ---------------------------------------------------------------------
# Helper: PNG → base64 string
# ---------------------------------------------------------------------

def encode_png_to_base64(png_ndarray):
    _, buf = cv2.imencode(".png", png_ndarray)
    return base64.b64encode(buf).decode()

# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=3000)
