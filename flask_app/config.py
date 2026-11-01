# config.py
import os

class Config:
    # Flask配置
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False

    # 摄像头配置
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30

    # 数据库配置
    DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'detections.db')

    # 视频流配置
    JPEG_QUALITY = 80
    STREAM_FPS = 20

    # 检测配置
    DETECTION_INTERVAL = 5  # 每5帧检测一次

config = Config()