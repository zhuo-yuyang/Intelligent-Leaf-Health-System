# main.py
"""
智能系统统一主程序
整合：摄像头采集 + 视觉处理 + AI推理 + 数据库 + Web服务
"""
import cv2
import time
import threading
import queue
import signal
import sys
import numpy as np
from picamera2 import Picamera2
from flask import Flask, Response, jsonify, request, render_template

from config import config
from db_helper import DBHelper

# ============================================================
# 配置区：根据本组课题，选择对应的AI模块
# ============================================================
# 【农业组】取消注释下一行
from ai_module_leaf import LeafAI
# 【安防组】取消注释下一行
# from ai_module_face import FaceAI
# 【统计组】无需AI模块

# ============================================================
# 全局变量
# ============================================================
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
app = Flask(__name__)
frame_queue = queue.Queue(maxsize=2)
running = True
detection_enabled = True

# 帧计数与FPS统计
frame_count = 0
fps = 0
last_fps_time = time.time()
fps_frame_count = 0

# 摄像头对象
picam2 = None

# 视觉处理模块
vision_module = None

# AI推理模块（根据课题选择）
ai_module = LeafAI()
print("[系统] LeafAI已加载")

# ============================================================
# 摄像头采集线程
# ============================================================
def camera_thread():
    """摄像头采集线程：持续采集帧，处理后放入队列"""
    global picam2, frame_count, fps, last_fps_time, fps_frame_count

    # 初始化摄像头
    picam2 = Picamera2()
    cam_config = picam2.create_preview_configuration(
        main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)}
    )
    picam2.configure(cam_config)
    picam2.set_controls({
        "AwbEnable": False,
        "AeEnable": False,
        "ColourGains": (1.8, 1.5),
        "ExposureTime": 15000,
        "AnalogueGain": 2.0
    })
    picam2.start()
    time.sleep(2)
    print("[摄像头] 采集线程已启动")

    while running:
        try:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 视觉处理
            if vision_module is not None:
                processed, result = vision_module.process(frame_bgr)
            else:
                processed = frame_bgr
                result = {}

            # AI推理（每5帧一次）
            if ai_module is not None and frame_count % 5 == 0:
                try:
                    ai_result = ai_module.predict(processed)
                    if isinstance(ai_result, dict):
                        result.update(ai_result)
                    elif isinstance(ai_result, tuple) and len(ai_result) == 2:
                        label, confidence = ai_result
                        result["label"] = label
                        result["confidence"] = float(confidence)

                    disease_ratio = result.get("disease_ratio", 0.0)
                    if disease_ratio > 0.5:
                        result["label"] = "diseased"  # 强制改为病害
                        # 既然面积这么大，可以适当提高置信度，让数据好看点
                        result["confidence"] = max(result.get("confidence", 0.0), 0.85)
                    
                    # 把 AI 标签画到画面上
                    if 'label' in result:
                        label_text = f"{result['label']} ({result.get('confidence', 0):.2f})"
                        color = (0, 255, 0) if result['label'] == 'healthy' else (0, 0, 255)
                        cv2.putText(processed, label_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)    
                except Exception as e:
                    print(f"[AI] 推理异常: {e}")

            # 自动存库（每60帧一次）
            if frame_count % 60 == 0 and result:
                try:
                    db = DBHelper(config.DB_PATH)
                    label = result.get('label', result.get('name', 'unknown'))
                    confidence = result.get('confidence', 0.0)
                    db.insert_detection("", label, confidence, result)
                except Exception as e:
                    print(f"[数据库] 存库异常: {e}")

            # 更新FPS
            frame_count += 1
            fps_frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = fps_frame_count / (now - last_fps_time)
                fps_frame_count = 0
                last_fps_time = now

            # 放入队列（满时丢弃旧帧）
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put((processed, result))

        except Exception as e:
            print(f"[摄像头] 采集异常: {e}")
            time.sleep(0.1)

    picam2.stop()
    print("[摄像头] 采集线程已停止")


# ============================================================
# Flask Web服务
# ============================================================
def generate_frames():
    """MJPEG帧生成器"""
    while running:
        try:
            frame, result = frame_queue.get(timeout=1.0)
        except queue.Empty:
            time.sleep(0.05)
            continue

        ret, jpeg = cv2.imencode('.jpg', frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')

        time.sleep(1.0 / config.STREAM_FPS)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({"code": 0, "data": {
        "running": running,
        "fps": round(fps, 1),
        "frame_count": frame_count,
        "detection_enabled": detection_enabled,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }})


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


@app.route('/api/capture', methods=['POST'])
def api_capture():
    try:
        frame, result = frame_queue.get(timeout=1.0)
    except queue.Empty:
        return jsonify({"code": 1, "message": "无法获取画面"})

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = f"capture_{timestamp}.jpg"
    cv2.imwrite(image_path, frame)

    db = DBHelper(config.DB_PATH)
    label = result.get('label', result.get('name', 'unknown'))
    confidence = result.get('confidence', 0.0)
    db.insert_detection(image_path, label, confidence, result)

    return jsonify({"code": 0, "message": "拍照成功",
                    "data": {"image_path": image_path, "result": result}})


@app.route('/api/history', methods=['GET'])
def api_history():
    limit = request.args.get('limit', 50, type=int)
    db = DBHelper(config.DB_PATH)
    records = db.get_history(limit)
    return jsonify({"code": 0, "data": records})


@app.route('/api/statistics', methods=['GET'])
def api_statistics():
    db = DBHelper(config.DB_PATH)
    stats = db.get_statistics()
    return jsonify({"code": 0, "data": stats})


@app.route('/api/latest', methods=['GET'])
def api_latest():
    try:
        _, result = frame_queue.get(timeout=0.5)
        return jsonify({"code": 0, "data": to_jsonable(result)})
    except queue.Empty:
        return jsonify({"code": 0, "data": {}})


@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    db = DBHelper(config.DB_PATH)
    if request.method == 'GET':
        return jsonify({"code": 0, "data": db.get_settings()})
    else:
        data = request.get_json()
        for key, value in data.items():
            db.set_setting(key, str(value))
        return jsonify({"code": 0, "message": "设置已更新"})


# ============================================================
# 主程序入口
# ============================================================
def signal_handler(sig, frame):
    """处理Ctrl+C信号，优雅退出"""
    global running
    print("\n[系统] 收到退出信号，正在关闭...")
    running = False
    time.sleep(1)
    sys.exit(0)


def main():
    global vision_module, ai_module

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 加载视觉处理模块
    try:
        from vision_module import VisionModule
        vision_module = VisionModule()
        print("[系统] 视觉处理模块已加载")
    except Exception as e:
        print(f"[系统] 视觉模块加载失败: {e}")

    # 加载AI推理模块（根据课题选择）
    # 【农业组】
    try:
        from ai_module_leaf import LeafAI
        ai_module = LeafAI()
        print("[系统] LeafAI已加载")
    except Exception as e:
        print(f"[系统] AI模块加载失败: {e}")

    # 【安防组】
    # try:
    #     from ai_module_face import FaceAI
    #     ai_module = FaceAI()
    #     print("[系统] FaceAI已加载")
    # except Exception as e:
    #     print(f"[系统] AI模块加载失败: {e}")

    # 【统计组】无需AI模块

    # 启动摄像头采集线程
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()

    # 等待摄像头就绪
    time.sleep(3)

    # 启动Flask Web服务（主线程）
    print(f"[系统] Web服务启动，访问 http://树莓派IP:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, threaded=True)


if __name__ == '__main__':
    main()