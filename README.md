# 智能植物叶片健康监测系统

基于树莓派 5 的边缘端叶片病害检测系统：摄像头实时采集 → HSV 分割与病斑占比计算 → MobileNetV2 分类 → SQLite 入库 → 浏览器可视化。

> 智能系统开发综合实训 课程项目　第 3 组
> 卓玉阳 · 陈炜 · 张佳明　指导教师：林灶钢

## 功能

- **实时视频流**：Picamera2 驱动 OV5647，640×480，MJPEG 推送，约 30 fps
- **视觉算法**：HSV 颜色空间分割叶片 → 形态学处理 → 最大轮廓筛选 → 病斑掩膜 → 病害面积占比
- **AI 分类**：MobileNetV2 迁移学习，TFLite 量化部署，单帧推理约 23.5 ms
- **数据存储**：SQLite 两张表（检测记录 / 系统设置），结果按时间间隔自动入库
- **终端控制页**：适配 3.5 寸 480×320 横屏触摸屏，含启动 / 停止 / 拍照按钮
- **管理后台**：统计卡片、健康分布饼图、健康病害对比柱状图、历史记录表、参数设置
- **开机自启**：systemd 托管，断电重启后自动恢复服务

## 系统架构

```
┌─ 应用层    Flask · HTML · ECharts   终端页 / 后台 / 8 个 API
├─ AI 推理层  TensorFlow Lite         MobileNetV2 → leaf_model.tflite
├─ 视觉算法层 OpenCV · NumPy           HSV 分割 · 形态学 · 病斑占比
└─ 感知层    Picamera2 · libcamera    OV5647 CSI · 640×480
```

## 目录结构

```
smart_system/
├── flask_app/
│   ├── main.py               # 统一主程序：采集线程 + Flask 服务
│   ├── config.py             # 配置（端口、分辨率、阈值）
│   ├── vision_module.py      # 视觉算法：HSV 分割与病斑占比
│   ├── ai_module_leaf.py     # TFLite 推理封装
│   ├── db_helper.py          # SQLite 操作与统计
│   ├── db_init.py            # 建表脚本
│   ├── camera_manager.py     # 摄像头管理（第五次课版本）
│   ├── app.py                # 早期 Flask 主程序（保留作参考）
│   ├── probe_labels.py       # 校验模型标签顺序
│   ├── clean_db.py           # 数据库清理与标签归一
│   ├── requirements.txt
│   ├── templates/
│   │   ├── index.html        # 终端控制界面
│   │   └── dashboard.html    # 管理后台
│   └── static/js/echarts.min.js
├── docs/
│   ├── 部署手册.md
│   └── 操作指南.md
├── smart_system.service      # systemd 服务定义
└── setup.sh                  # 环境一键重建脚本
```

## 环境要求

| 项目 | 版本 / 配置 |
|:---|:---|
| 硬件 | Raspberry Pi 5（4 GB）、OV5647 CSI 摄像头、MHS-3.5inch 触摸屏 |
| 系统 | Raspberry Pi OS 64 位（Debian 12/13） |
| Python | 3.13，虚拟环境需带 `--system-site-packages` |
| 主要依赖 | OpenCV 4.10、NumPy、TensorFlow Lite、Flask 3.1 |

## 快速开始

```bash
# 1. 系统依赖（picamera2 只能通过 apt 安装）
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-venv python3-pip

# 2. 创建虚拟环境（必须带 --system-site-packages）
cd ~/smart_system
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r flask_app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 初始化数据库并启动
cd flask_app
python db_init.py
python main.py
```

浏览器访问 `http://树莓派IP:5000/`（终端控制页）与 `/dashboard`（管理后台）。

## 接口一览

| 接口 | 方法 | 功能 |
|:---|:---|:---|
| `/video_feed` | GET | MJPEG 视频流 |
| `/api/status` | GET | 系统状态（FPS、帧数） |
| `/api/start` / `/api/stop` | POST | 启动 / 停止检测 |
| `/api/capture` | POST | 拍照并存库 |
| `/api/history` | GET | 历史记录 |
| `/api/statistics` | GET | 统计（健康/病害/未知计数） |
| `/api/latest` | GET | 最新检测结果 |
| `/api/settings` | GET/POST | 读取 / 更新设置 |

## 已知问题与改进方向

- 病害判定阈值对光照敏感，强光 / 遮挡场景下结果会退化为 `unknown`
- 模型训练集仅 92 张（健康 46 / 病害 46），泛化能力有限
- `detail_info` 字段曾把整幅掩膜数组写入数据库导致体积膨胀，现已改为只存病害占比等摘要字段
- 后续计划：YOLOv8n 多叶片实例分割、病斑分级、MQTT 上云

## 说明

`detections.db`、`*.tflite`、`venv/` 等大文件已在 `.gitignore` 中排除，需单独分发。
模型文件 `leaf_model.tflite` 请向组员索取。
