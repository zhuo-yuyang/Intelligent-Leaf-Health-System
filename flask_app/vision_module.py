# vision_module.py (农业组)
import cv2
import numpy as np

class VisionModule:
    def __init__(self):
        self.params = {
            "lower_green": np.array([0, 40, 40]),
            "upper_green": np.array([57, 255, 255]),
            "lower_disease": np.array([10, 60, 40]),
            "upper_disease": np.array([25, 255, 255])
        }

    def process(self, frame):
        """统一接口：返回(processed_frame, result_dict)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        leaf_mask = cv2.inRange(hsv, self.params["lower_green"],
                                self.params["upper_green"])
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel_small)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_large)

        disease_mask = cv2.inRange(hsv, self.params["lower_disease"],
                                   self.params["upper_disease"])
        disease_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=leaf_mask)

        leaf_area = np.sum(leaf_mask > 0)
        disease_area = np.sum(disease_mask > 0)
        ratio = disease_area / leaf_area if leaf_area > 0 else 0

        # ===== 新增：把 mask 可视化到帧上 =====
        processed = frame.copy()

        # 1) 找叶片轮廓，画绿色矩形框
        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(processed, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
                cv2.putText(processed, "Leaf", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 2) 病斑区域红色叠加
        processed[disease_mask > 0] = [0, 0, 255]

        # 3) 左上角显示病害占比
        cv2.putText(processed, f"Disease: {ratio*100:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return processed, {   # ← 返回画好的帧
            "leaf_mask": leaf_mask,
            "disease_mask": disease_mask,
            "disease_ratio": ratio
        }