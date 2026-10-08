import os
import io
from flask import Flask, request, jsonify
from PIL import Image
from google import genai
from google.genai import types

app = Flask(__name__)

# Khởi tạo Gemini Client (Đảm bảo bạn đã set biến môi trường GEMINI_API_KEY hoặc thay trực tiếp khóa API vào đây)
# client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


@app.route('/classify', methods=['POST'])
def classify_trash():
    """
    API endpoint này sẽ nhận ảnh từ ESP32-CAM (hoặc từ phần mềm test gửi lên)
    """
    if 'file' not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Chưa chọn file ảnh"}), 400

    try:
        # Đọc dữ liệu ảnh nhận được
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Prompt yêu cầu AI nhận diện và trả về đúng định dạng yêu cầu cho thùng rác
        prompt = (
            "Hãy phân tích bức ảnh này và xác định vật thể trong ảnh thuộc loại rác nào sau đây: "
            "'paper' (giấy), 'plastic' (nhựa), hoặc 'metal' (kim loại). "
            "Bạn CHỈ ĐƯỢC PHÉP trả về đúng một từ duy nhất: 'paper', 'plastic', hoặc 'metal' mà không kèm theo bất kỳ giải thích nào khác."
        )

        # Gọi Gemini Model với cấu hình tắt tool/function calling để tránh lỗi xung đột khi xử lý ảnh
        response = client.models.generate_content(
            model='gemini-flash-latest',  # Đổi sang gemini-2.5-flash
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                tools=[]
            )
        )

        # Lấy kết quả trả về từ AI và làm sạch khoảng trắng nếu có
        result_class = response.text.strip().lower()

        # Đảm bảo kết quả nằm trong 3 loại hợp lệ
        if result_class not in ['paper', 'plastic', 'metal']:
            result_class = 'unknown'

        print(f"-> AI nhận diện được: {result_class}")

        # Trả kết quả về dạng JSON cho ESP32-CAM đọc sau này
        return jsonify({
            "status": "success",
            "prediction": result_class
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    # Chạy server ở cổng 5000, lắng nghe từ mọi địa chỉ mạng để sau này ESP32 kết nối vào IP máy tính
    app.run(host='0.0.0.0', port=5000, debug=True)