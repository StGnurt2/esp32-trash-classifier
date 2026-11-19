import os
import base64
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

app = Flask(__name__)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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
    "updated_at": "Chưa có ảnh"
}

capture_requested = False

def classify_trash_jpeg(jpeg_bytes: bytes) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                PROMPT,
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        raw = (response.text or "").strip().lower()
        print(f"Gemini raw response: {raw!r}", flush=True)
        for label in VALID_LABELS:
            if raw == label or raw.startswith(label):
                return label
        return "unknown"
    except Exception as e:
        print(f"Gemini error: {repr(e)}", flush=True)
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
    img_bytes = request.get_data()
    print(f"Received /predict bytes: {len(img_bytes)}", flush=True)
    if not img_bytes:
        return jsonify({"error": "No image", "label": "unknown"}), 400

@app.route("/api/latest", methods=["GET"])
def api_latest():
    return jsonify(latest_result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
