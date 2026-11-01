#!/bin/bash
# setup.sh - 智能视觉系统环境重建脚本
set -e
echo "===== 1. 更新系统 ====="
sudo apt update
echo "===== 2. 安装系统级依赖 ====="
sudo apt install -y python3-picamera2 python3-libcamera libcamera-apps python3-venv python3-pip git
echo "===== 3. 创建虚拟环境（必须加 --system-site-packages）====="
cd ~/smart_system
python3 -m venv --system-site-packages venv
source venv/bin/activate
echo "===== 4. 安装 pip 依赖 ====="
pip install --upgrade pip
pip install -r flask_app/requirements.txt -i https://pypi.tuna.tsin
ghua.edu.cn/simple
echo "===== 5. 验证 ====="
python -c "from picamera2 import Picamera2; print('picamera2 OK')"
python -c "import cv2; print('opencv', cv2.__version__)"
python -c "import flask; print('flask', flask.__version__)"
echo "===== 环境重建完成 ====="
赋予执行权限：
bash
chmod +x ~/smart_system/setup.sh
