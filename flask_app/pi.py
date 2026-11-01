from db_helper import DBHelper
from config import config

db = DBHelper(config.DB_PATH)
did = db.insert_detection("test_code.jpg", "healthy", 0.95, {"label": "healthy"})
print("插入ID:", did)
print(db.get_history(5))