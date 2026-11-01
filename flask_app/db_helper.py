# db_helper.py
import numpy as np
import sqlite3
import json


def _to_python(obj):
    """递归把 numpy 类型转成原生 Python 类型"""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class DBHelper:
    def __init__(self, db_path="detections.db"):
        self.db_path = db_path

    def insert_detection(self, image_path, result_label, confidence=0.0, detail_info=None):
        if hasattr(confidence, 'item'):
            confidence = float(confidence)
        detail_info = _to_python(detail_info) if detail_info else None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detections (image_path, result_label, confidence, detail_info, source)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            str(image_path),
            str(result_label),
            confidence,
            json.dumps(detail_info, ensure_ascii=False) if detail_info else None,
            "live"
        ))
        conn.commit()
        detection_id = cursor.lastrowid
        conn.close()
        return detection_id

    def get_history(self, limit=100):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, timestamp, image_path, result_label, confidence
            FROM detections ORDER BY id DESC LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_statistics(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM detections')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT result_label, COUNT(*) FROM detections GROUP BY result_label')
        by_label = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) FROM detections WHERE DATE(timestamp)=DATE('now')")
        today = cursor.fetchone()[0]
        conn.close()
        return {"total": total, "by_label": dict(by_label), "today": today}
    
    def get_settings(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT key, value FROM system_settings')
        rows = cursor.fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}

    def set_setting(self, key, value):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO system_settings (key, value) VALUES (?, ?)
        ''', (key, value))
        conn.commit()
        conn.close()