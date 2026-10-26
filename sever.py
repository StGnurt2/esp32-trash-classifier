import os
import io
import gc
from flask import Flask, request, jsonify, send_file
from PIL import Image
from google import genai
from google.genai import types

app = Flask(__name__)

# Khởi tạo Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Đường dẫn lưu ảnh tạm thời trên server
LATEST_IMAGE_PATH = "latest_image.jpg"

# Cờ lệnh điều khiển và biến lưu kết quả
should_capture = False
latest_result = {"status": "waiting", "prediction": "Chưa có dữ liệu"}

@app.route('/', methods=['GET'])
def home():
    """Giao diện điều khiển nút bấm IoT trên Điện thoại / Máy tính"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hệ Thống Phân Loại Rác Smart IoT</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { text-align: center; font-family: Arial, sans-serif; background-color: #f4f4f9; padding: 20px; }
            .card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); max-width: 480px; margin: auto; }
            img { max-width: 100%; height: auto; border-radius: 8px; border: 2px solid #ddd; margin-top: 15px; }
            button { background-color: #007bff; color: white; border: none; padding: 15px 30px; font-size: 18px; font-weight: bold; border-radius: 8px; cursor: pointer; transition: 0.2s; width: 100%; }
            button:hover { background-color: #0056b3; }
            button:disabled { background-color: #6c757d; cursor: not-allowed; }
            .result { margin-top: 20px; font-size: 20px; font-weight: bold; color: #28a745; background: #e9f7ef; padding: 10px; border-radius: 6px; }
            .status { margin-top: 10px; font-size: 14px; color: #666; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Phân Loại Rác Thông Minh</h2>
            <button id="btnCapture" onclick="triggerCapture()">📸 CHỤP & PHÂN LOẠI RÁC</button>
            <div id="statusText" class="status">Trạng thái: Sẵn sàng nhận lệnh</div>
            <div id="resultText" class="result">Chưa phân loại</div>
            <img id="camImg" src="/image" alt="Đang tải ảnh...">
        </div>

        <script>
            let pollingTimer = null;

            function triggerCapture() {
                const btn = document.getElementById('btnCapture');
                const status = document.getElementById('statusText');
                const result = document.getElementById('resultText');

                btn.disabled = true;
                status.innerText = "Đã gửi lệnh! Đang chờ ESP32-CAM chụp...";
                result.innerText = "Đang xử lý...";

                fetch('/trigger-capture', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    startPollingResult();
                })
                .catch(err => {
                    btn.disabled = false;
                    status.innerText = "Lỗi kết nối Server!";
                });
            }

            function startPollingResult() {
                if (pollingTimer) clearInterval(pollingTimer);
                
                pollingTimer = setInterval(() => {
                    fetch('/get-result')
                    .then(res => res.json())
                    .then(data => {
                        if (data.status === 'success') {
                            document.getElementById('resultText').innerText = "Loại rác: " + data.prediction.toUpperCase();
                            document.getElementById('statusText').innerText = "Trạng thái: Hoàn thành!";
                            document.getElementById('camImg').src = '/image?t=' + new Date().getTime();
                            document.getElementById('btnCapture').disabled = false;
                            clearInterval(pollingTimer);
                        } else if (data.status === 'error') {
                            document.getElementById('resultText').innerText = "Lỗi: " + data.prediction;
                            document.getElementById('btnCapture').disabled = false;
                            clearInterval(pollingTimer);
                        }
                    });
                }, 1000);
            }
        </script>
    </body>
    </html>
    """
    return html

@app.route('/trigger-capture', methods=['POST'])
def trigger_capture():
    global should_capture, latest_result
    should_capture = True
    latest_result = {"status": "processing", "prediction": "Đang xử lý..."}
    return jsonify({"status": "triggered"}), 200

@app.route('/check-trigger', methods=['GET'])
def check_trigger():
    global should_capture
    if should_capture:
        should_capture = False  # Reset lại cờ sau khi ESP32 đã đọc lệnh
        return jsonify({"capture": True}), 200
    return jsonify({"capture": False}), 200

@app.route('/get-result', methods=['GET'])
def get_result():
    return jsonify(latest_result), 200

@app.route('/image', methods=['GET'])
def get_image():
    if os.path.exists(LATEST_IMAGE_PATH):
        return send_file(LATEST_IMAGE_PATH, mimetype='image/jpeg')
    return "Chưa có ảnh", 404

@app.route('/classify', methods=['POST'])
def classify_trash():
    global latest_result
    if 'file' not in request.files:
        latest_result = {"status": "error", "prediction": "No file"}
        return jsonify(latest_result), 400

    file = request.files['file']
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image.save(LATEST_IMAGE_PATH)

        prompt = (
            "Hãy phân tích bức ảnh này và xác định vật thể rác thuộc loại nào sau đây: "
            "'paper' (giấy), 'plastic' (nhựa), hoặc 'metal' (kim loại). "
            "CHỈ TRẢ VỀ ĐÚNG 1 TỪ DUY NHẤT: 'paper', 'plastic', hoặc 'metal'."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt],
            config=types.GenerateContentConfig(temperature=0.1)
        )

        result_class = response.text.strip().lower()
        if result_class not in ['paper', 'plastic', 'metal']:
            result_class = 'unknown'

        del image_bytes
        del image
        gc.collect()

        latest_result = {"status": "success", "prediction": result_class}
        return jsonify(latest_result), 200

    except Exception as e:
        latest_result = {"status": "error", "prediction": str(e)}
        return jsonify(latest_result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
