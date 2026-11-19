import io
import os
import base64
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from PIL import Image
from google import genai
from google.genai import types

app = Flask(__name__)

API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()
client = genai.Client(api_key=API_KEY) if API_KEY else None

VALID_LABELS = ("paper", "glass", "plastic")

PROMPT = """
Nhìn ảnh rác này và phân loại chính xác vào ĐÚNG MỘT trong ba nhãn sau:
- paper (giấy, bìa carton, giấy vở, khăn giấy)
- glass (thủy tinh, chai lọ thủy tinh, mảnh kính)
- plastic (nhựa, chai nhựa, hộp nhựa, túi nilon)

Chỉ trả về DUY NHẤT một từ viết thường: paper, glass, hoặc plastic. Không giải thích.
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
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    return None


def classify_trash_jpeg(jpeg_bytes: bytes) -> str:
    if not client:
        print("Gemini error: GEMINI_API_KEY is missing or empty", flush=True)
        return "unknown"

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                PROMPT,
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        )

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
    except Exception as e:
        print(f"Gemini error: {repr(e)}", flush=True)
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
                print("Converted HQVGA grayscale frame to JPEG", flush=True)
            else:
                print("Warning: received non-JPEG data with unexpected length", flush=True)

        label = classify_trash_jpeg(img_bytes)
        updated_at = datetime.now().strftime("%H:%M:%S")

        latest_result["label"] = label
        latest_result["image_b64"] = base64.b64encode(img_bytes).decode("utf-8")
        latest_result["updated_at"] = updated_at

        return jsonify({"label": label, "updated_at": updated_at}), 200

    except Exception as e:
        print(f"Predict route error: {repr(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        updated_at = datetime.now().strftime("%H:%M:%S")
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
