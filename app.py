import os
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# Khởi tạo client (tự đọc biến môi trường GEMINI_API_KEY trên Render hoặc máy tính)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

VALID_LABELS = ("paper", "glass", "plastic")

PROMPT = """
Hãy nhìn bức ảnh rác này và phân loại chính xác vào ĐÚNG MỘT trong ba nhãn sau:
- paper (giấy, bìa carton, giấy vở, khăn giấy)
- glass (thủy tinh, chai lọ thủy tinh, mảnh kính)
- plastic (nhựa, chai nhựa, hộp nhựa, túi nilon)

Chỉ trả về DUY NHẤT một từ viết thường nằm trong danh sách: paper, glass, hoặc plastic. Không giải thích, không thêm dấu câu.
""".strip()

def classify_trash_jpeg(jpeg_bytes: bytes) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=jpeg_bytes,
                    mime_type="image/jpeg",
                ),
                PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )
        raw_text = (response.text or "").strip().lower()
        for label in VALID_LABELS:
            if raw_text == label or raw_text.startswith(label):
                return label
        return "unknown"
    except Exception as e:
        print("Gemini error:", e)
        return "unknown"

@app.route("/predict", methods=["POST"])
def predict():
    img_bytes = request.get_data()
    if not img_bytes:
        return jsonify({"error": "No image data", "label": "unknown"}), 400

    label = classify_trash_jpeg(img_bytes)
    return jsonify({"label": label})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
