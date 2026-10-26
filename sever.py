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

@app.route('/', methods=['GET'])
def home():
    """Trang chủ hiển thị giao diện xem ảnh (Dùng JavaScript tải ảnh mượt hơn, không reload toàn trang)"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ESP32-CAM Viewer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { text-align: center; font-family: Arial; background-color: #f4f4f9; padding-top: 20px; }
            img { max-width: 90%; max-height: 70vh; border: 4px solid #333; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
            h2 { color: #333; }
            p { color: #666; font-size: 14px; }
        </style>
    </head>
    <body>
        <h2>Ảnh mới nhất từ ESP32-CAM</h2>
        <p>Ảnh sẽ tự động làm mới mỗi 5 giây</p>
        <img id="camImg" src="/image" alt="Đang chờ ESP32-CAM gửi ảnh lên...">
        <script>
            // Reload ảnh bằng JS tránh crash/nghẽn server
            setInterval(function() {
                var img = document.getElementById('camImg');
                img.src = '/image?t=' + new Date().getTime();
            }, 5000);
        </script>
    </body>
    </html>
    """
    return html

@app.route('/image', methods=['GET'])
def get_image():
    """API trả về file ảnh mới nhất để hiển thị lên web"""
    if os.path.exists(LATEST_IMAGE_PATH):
        return send_file(LATEST_IMAGE_PATH, mimetype='image/jpeg')
    else:
        return "Chưa có ảnh", 404

@app.route('/classify', methods=['POST'])
def classify_trash():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "Không tìm thấy file ảnh"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Chưa chọn file ảnh"}), 400

    try:
        # 1. Đọc dữ liệu ảnh nhận được
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 2. LƯU LẠI ẢNH ĐỂ HIỂN THỊ LÊN TRANG CHỦ
        image.save(LATEST_IMAGE_PATH)

        # 3. Gửi cho AI phân tích
        prompt = (
            "Hãy phân tích bức ảnh này và xác định vật thể trong ảnh thuộc loại rác nào sau đây: "
            "'paper' (giấy), 'plastic' (nhựa), hoặc 'metal' (kim loại). "
            "Bạn CHỈ ĐƯỢC PHÉP trả về đúng một từ duy nhất: 'paper', 'plastic', hoặc 'metal' mà không kèm theo bất kỳ giải thích nào khác."
        )

        # SỬA TÊN MODEL THÀNH gemini-2.5-flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt],
            config=types.GenerateContentConfig(temperature=0.1)
        )

        result_class = response.text.strip().lower()
        if result_class not in ['paper', 'plastic', 'metal']:
            result_class = 'unknown'

        # Thu dọn rác bộ nhớ RAM
        del image_bytes
        del image
        gc.collect()

        return jsonify({
            "status": "success",
            "prediction": result_class
        }), 200

    except Exception as e:
        print(f"Lỗi Server: {str(e)}")
        # Ép luôn trả về JSON khi gặp exception để ESP32 không bị dính HTML error
        return jsonify({"status": "error", "message": str(e)}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
