# db_init.py
import sqlite3

def init_db(db_path="detections.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT,
            result_label TEXT NOT NULL,
            confidence REAL,
            detail_info TEXT,
            source TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print(f"数据库 {db_path} 初始化完成")

if __name__ == "__main__":
    init_db()