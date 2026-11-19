import io
import os
import time
import base64
import traceback
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify, render_template
from PIL import Image
from google import genai
from google.genai import types

VN_TZ = timezone(timedelta(hours=7))

app = Flask(__name__)

API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()
client = genai.Client(api_key=API_KEY) if API_KEY else None

MODEL_NAME = (os.environ.get("GEMINI_MODEL") or "gemini-3.8-flash").strip()
FALLBACK_MODEL_NAME = (os.environ.get("GEMINI_FALLBACK_MODEL") or "").strip()

VALID_LABELS = ("paper", "glass", "plastic")

PROMPT = """
Phân loại vật thể rác chính trong ảnh vào ĐÚNG MỘT từ viết thường: paper, glass, hoặc plastic.
- paper: giấy, bìa carton, vở, hộp giấy
- glass: thủy tinh, chai/lọ/hũ thủy tinh, kính
- plastic: nhựa, chai nhựa, hộp nhựa, túi nilon
Chỉ trả về đúng 1 từ: paper, glass, hoặc plastic (chỉ trả unknown nếu hoàn toàn không có vật thể nào).
""".strip()

latest_result = {
    "label": "none",
    "image_b64": "",
    "updated_at": "Chưa có ảnh",
}

capture_requested = False


def is_valid_jpeg(data: bytes) -> bool:
    return bool(data) and len(data) > 4 and data[0:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


def convert_raw_to_jpeg(raw_bytes: bytes) -> bytes | None:
    if len(raw_bytes) == 240 * 176:
        img = Image.frombytes("L", (240, 176), raw_bytes)
        # Chỉ bật img = img.rotate(180) ở đây nếu CHƯA lật 180 độ trên ESP32-CAM (.ino)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70, optimize=True)
        return buf.getvalue()
    return None


def _build_http_options(timeout_ms: int) -> types.HttpOptions:
    try:
        return types.HttpOptions(
            timeout=timeout_ms,
            retry=types.HttpRetryOptions(attempts=1),
        )
    except Exception:
        return types.HttpOptions(timeout=timeout_ms)


def _call_model_once(model_id: str, jpeg_bytes: bytes, timeout_ms: int = 12000) -> str:
    t0 = time.time()
    print(f"Calling Gemini ({model_id}, timeout={timeout_ms}ms)...", flush=True)
    response = client.models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            http_options=_build_http_options(timeout_ms),
        ),
    )
    print(f"Gemini ({model_id}) responded in {time.time() - t0:.2f}s", flush=True)

    text = getattr(response, "text", None)
    raw = (text or "").strip().lower()
    print(f"Gemini raw response: {raw!r}", flush=True)

    for label in VALID_LABELS:
        if raw == label or raw.startswith(label):
            return label

    for label in VALID_LABELS:
        if label in raw:
            return label

    return "unknown"


def classify_trash_jpeg(jpeg_bytes: bytes) -> str:
    if not client:
        print("Gemini error: GEMINI_API_KEY is missing or empty", flush=True)
        return "unknown"

    try:
        return _call_model_once(MODEL_NAME, jpeg_bytes, timeout_ms=12000)
    except Exception as e:
        err_str = repr(e)
        print(f"Gemini error ({MODEL_NAME}): {err_str}", flush=True)
        is_timeout_or_busy = (
            "504" in err_str
            or "503" in err_str
            or "DEADLINE_EXCEEDED" in err_str
            or "timed out" in err_str.lower()
            or "timeout" in err_str.lower()
        )
        if FALLBACK_MODEL_NAME and FALLBACK_MODEL_NAME != MODEL_NAME and is_timeout_or_busy:
            try:
                print(f"Switching fast to fallback model: {FALLBACK_MODEL_NAME}", flush=True)
                return _call_model_once(FALLBACK_MODEL_NAME, jpeg_bytes, timeout_ms=20000)
            except Exception as fb_err:
                print(f"Gemini fallback error ({FALLBACK_MODEL_NAME}): {repr(fb_err)}", flush=True)
                print(traceback.format_exc(), flush=True)
                return "unknown"
        print(traceback.format_exc(), flush=True)
        return "unknown"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/trigger", methods=["POST"])
def api_trigger():
    global capture_requested
    capture_requested = True
    return jsonify({"status": "queued", "capture": True})


@app.route("/api/should_capture", methods=["GET"])
def api_should_capture():
    global capture_requested
    should_now = capture_requested
    if capture_requested:
        capture_requested = False
    return jsonify({"capture": should_now})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_bytes = request.get_data() or b""
        print(f"Received /predict bytes: {len(img_bytes)}", flush=True)

        if not img_bytes:
            return jsonify({"error": "No image", "label": "unknown"}), 400

        if not is_valid_jpeg(img_bytes):
            converted = convert_raw_to_jpeg(img_bytes)
            if converted:
                img_bytes = converted
                print(f"Converted HQVGA grayscale frame to JPEG ({len(img_bytes)} bytes)", flush=True)
            else:
                print("Warning: received non-JPEG data with unexpected length", flush=True)

        label = classify_trash_jpeg(img_bytes)
        updated_at = datetime.now(VN_TZ).strftime("%H:%M:%S")

        latest_result["label"] = label
        latest_result["image_b64"] = base64.b64encode(img_bytes).decode("utf-8")
        latest_result["updated_at"] = updated_at

        return jsonify({"label": label, "updated_at": updated_at}), 200

    except Exception as e:
        print(f"Predict route error: {repr(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        updated_at = datetime.now(VN_TZ).strftime("%H:%M:%S")
        latest_result["label"] = "unknown"
        latest_result["updated_at"] = updated_at
        return jsonify({
            "label": "unknown",
            "updated_at": updated_at,
            "error": "Internal processing error",
        }), 200


@app.route("/api/latest", methods=["GET"])
def api_latest():
    return jsonify(latest_result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
