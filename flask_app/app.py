# app.py
from flask import Flask, Response, jsonify, request
import cv2
import json
import time
from config import config
from camera_manager import CameraManager
from ai_module_leaf import LeafAI
import numpy as np
from flask import render_template

app = Flask(__name__)
ai = LeafAI() 

# 全局摄像头管理器
camera = CameraManager(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
camera.set_ai_module(ai)
print(">>> AI 模块加载成功（LeafAI）")
# camera.start()
# 全局检测开关
detection_enabled = True
latest_result = {}

def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        # ★ 不返回完整数组，只返回形状信息
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# === 新增：结果融合逻辑 ===
def merge_results(ai_result, vision_result):
    """
    将 AI 分类结果和传统视觉计算出的病害率进行融合
    """
    if not ai_result:
        return None
        
    final_result = ai_result.copy()
    
    # 获取视觉模块算出的病害率，如果没有就算 0
    disease_ratio = vision_result.get('disease_ratio', 0) if vision_result else 0
    final_result['disease_ratio'] = disease_ratio
    
    # 核心逻辑：如果病害面积占比超过 50%，强制覆盖 AI 的结论
    if disease_ratio > 0.5:
        final_result['label'] = 'diseased'  # 强制改为病害
        # 既然视觉算出大面积病害，我们可以适当提升一点置信度
        final_result['confidence'] = max(final_result.get('confidence', 0.0), 0.85) 
        
    return final_result

def generate_frames():
    """MJPEG帧生成器"""
    while True:
        frame, result = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        # 编码为JPEG
        ret, jpeg = cv2.imencode('.jpg', frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        if not ret:
            continue

        # 以MJPEG格式推送
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')

        # 控制推送帧率
        time.sleep(1.0 / config.STREAM_FPS)


@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ========== 系统状态 ==========
@app.route('/api/status', methods=['GET'])
def api_status():
    status = camera.get_status()
    status['detection_enabled'] = detection_enabled
    status['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({"code": 0, "data": status})


# ========== 检测控制 ==========
@app.route('/api/start', methods=['POST'])
def api_start():
    global detection_enabled
    detection_enabled = True
    return jsonify({"code": 0, "message": "检测已启动"})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    global detection_enabled
    detection_enabled = False
    return jsonify({"code": 0, "message": "检测已停止"})


# ========== 手动拍照 ==========
@app.route('/api/capture', methods=['POST'])
def api_capture():
    frame, result = camera.get_frame()
    if frame is None:
        return jsonify({"code": 1, "message": "无法获取画面"})

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = f"capture_{timestamp}.jpg"
    cv2.imwrite(image_path, frame)

    from db_helper import DBHelper
    db = DBHelper(config.DB_PATH)
    label = result.get('label', 'unknown') if result else 'unknown'
    confidence = result.get('confidence', 0.0) if result else 0.0
    db.insert_detection(image_path, label, confidence, result)

    if result:
        result_summary = {k: v for k, v in result.items()
                          if not isinstance(v, np.ndarray)}
    else:
        result_summary = None

    return jsonify({
        "code": 0,
        "message": "拍照成功",
        "data": {"image_path": image_path, "result": to_jsonable(result_summary)}
    })


# ========== 历史记录 ==========
@app.route('/api/history', methods=['GET'])
def api_history():
    limit = request.args.get('limit', 50, type=int)
    from db_helper import DBHelper
    db = DBHelper(config.DB_PATH)
    records = db.get_history(limit)
    return jsonify({"code": 0, "data": records})


# ========== 统计数据 ==========
@app.route('/api/statistics', methods=['GET'])
def api_statistics():
    from db_helper import DBHelper
    db = DBHelper(config.DB_PATH)
    stats = db.get_statistics()
    return jsonify({"code": 0, "data": stats})


# ========== 最新检测结果 ==========
@app.route('/api/latest', methods=['GET'])
def api_latest():
    frame, result = camera.get_frame()
    if result:
        result = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}

        disease_ratio = result.get('disease_ratio', 0)
        if disease_ratio > 0.5:
            result['label'] = 'diseased'
            
    return jsonify({"code": 0, "data": to_jsonable(result) if result else {}})

# ========== 系统设置 ==========
@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    from db_helper import DBHelper
    db = DBHelper(config.DB_PATH)

    if request.method == 'GET':
        settings = db.get_settings()
        return jsonify({"code": 0, "data": settings})
    else:
        data = request.get_json()
        for key, value in data.items():
            db.set_setting(key, str(value))
        return jsonify({"code": 0, "message": "设置已更新"})


# ========== 主页 ==========
@app.route('/')
def index():
    return render_template('index.html')
# def index():
#     return jsonify({
#         "code": 0,
#         "message": "智能系统后端服务",
#         "endpoints": [
#             "/video_feed", "/api/status", "/api/start", "/api/stop",
#             "/api/capture", "/api/history", "/api/statistics",
#             "/api/latest", "/api/settings"
#         ]
#     })

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# ========== 启动 ==========
if __name__ == '__main__':
    try:
        from vision_module import VisionModule
        vision = VisionModule()
        camera.set_vision_module(vision)
    except Exception as e:
        print(f"视觉模块加载失败: {e}")

    camera.start()
    app.run(host=config.HOST, port=config.PORT, threaded=True)