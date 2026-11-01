# camera_manager.py
import cv2
import time
import threading
import queue
import numpy as np
from picamera2 import Picamera2

class CameraManager:
    """摄像头管理类，负责采集、处理、分发帧"""

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.picam2 = None

        # 处理模块（由外部注入）
        self.vision_module = None
        self.ai_module = None

        # 统计信息
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.last_label = None
        self.last_confidence = 0.0

    def set_vision_module(self, vision_module):
        """注入视觉处理模块"""
        self.vision_module = vision_module

    def set_ai_module(self, ai_module):
        """注入AI推理模块"""
        self.ai_module = ai_module

    def start(self):
        """启动摄像头采集线程"""
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (self.width, self.height)}
        )
        self.picam2.configure(config)
        self.picam2.set_controls({
            "AwbEnable": False,
            "AeEnable": False,
            "ColourGains": (1.8, 1.5),
            "ExposureTime": 15000,
            "AnalogueGain": 2.0
        })
        self.picam2.start()
        time.sleep(2)

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("摄像头采集线程已启动")

    def _capture_loop(self):
        """采集循环（在独立线程中运行）"""
        while self.running:
            try:
                
                frame_rgb = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # 更新FPS
                self.frame_count += 1
                self.fps_frame_count += 1
                now = time.time()

                # 视觉处理
                if self.vision_module is not None:
                    processed, result = self.vision_module.process(frame_bgr)
                else:
                    processed = frame_bgr
                    result = {}

                # 每5帧执行一次AI推理
                if self.ai_module is not None and self.frame_count % 5 == 0:
                    try:
                        ai_result = self.ai_module.predict(processed)

                        if isinstance(ai_result, dict):
                            result.update(ai_result)
                        elif isinstance(ai_result, tuple) and len(ai_result) == 2:
                            label, confidence = ai_result
                            result["label"] = label
                            result["confidence"] = float(confidence)
                            self.last_label = label
                            self.last_confidence = float(confidence)

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
                            result["confidence"] = self.last_confidence
                    except Exception as e:
                        print(f"AI推理异常: {e}")

                
                # 在_capture_loop中添加
                if self.frame_count % 30 == 0 and result:
                    try:
                        from db_helper import DBHelper
                        from config import config
                        db = DBHelper(config.DB_PATH)
                        db.insert_detection("", result.get('label', 'unknown'),
                             result.get('confidence', 0.0), result)
                        print(f"[DB] 已存库: {result.get('label')} "
                              f"({result.get('confidence', 0):.2f})")
                    except Exception as e:
                        print(f"存库异常: {e}")

                
                if now - self.last_fps_time >= 1.0:
                    self.fps = self.fps_frame_count / (now - self.last_fps_time)
                    self.fps_frame_count = 0
                    self.last_fps_time = now

                # 放入队列（如果队列满，丢弃旧帧）
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put((processed, result))

                

            except Exception as e:
                print(f"采集异常: {e}")
                time.sleep(0.1)

    def get_frame(self):
        """获取最新帧"""
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None, None

    def stop(self):
        """停止采集"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.picam2 is not None:
            self.picam2.stop()
        print("摄像头采集线程已停止")

    def get_status(self):
        """获取摄像头状态"""
        return {
            "running": self.running,
            "fps": round(self.fps, 1),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height
        }