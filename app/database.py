from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# ใช้ SQLite ในการเก็บข้อมูล (ง่าย ไม่ต้องตั้งค่าเซิร์ฟเวอร์แยก)
SQLALCHEMY_DATABASE_URL = "sqlite:///./plant_health.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency สำหรับให้ API ดึง Database Session ไปใช้งาน
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()