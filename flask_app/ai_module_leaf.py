# ai_module_leaf.py
import os
import cv2
import numpy as np
from tensorflow.lite.python import interpreter as tflite
# import tflite_runtime.interpreter as tflite

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class LeafAI:
    def __init__(self, model_path=None):
        if model_path is None:
            # 默认模型路径：与本脚本同目录的 leaf_model.tflite
            model_path = os.path.join(BASE_DIR, "leaf_model.tflite")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = ["healthy", "diseased"]
    
    def predict(self, image):
        """输入BGR图像（224x224），返回标签和置信度"""
        img = cv2.resize(image, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 127.5 - 1.0
        img = np.expand_dims(img, axis=0)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        pred_idx = np.argmax(output)
        confidence = float(output[0][pred_idx])
        return self.labels[pred_idx], confidence
