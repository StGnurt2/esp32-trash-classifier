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
        for label in VALID_LABELS:
            if raw == label or raw.startswith(label):
                return label
        return "unknown"
    except Exception as e:
        print("Gemini error:", e)
        return "unknown"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    img_bytes = request.get_data()
    if not img_bytes:
        return jsonify({"error": "No image", "label": "unknown"}), 400

    label = classify_trash_jpeg(img_bytes)
    latest_result["label"] = label
    latest_result["image_b64"] = base64.b64encode(img_bytes).decode("utf-8")
    latest_result["updated_at"] = datetime.now().strftime("%H:%M:%S")

    return jsonify({"label": label, "updated_at": latest_result["updated_at"]})

@app.route("/api/latest", methods=["GET"])
def api_latest():
    return jsonify(latest_result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
