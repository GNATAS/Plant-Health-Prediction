from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from .database import engine, Base
from .routers import fields, predictions

# 🌟 คำสั่งนี้จะสร้างตารางในไฟล์ SQLite อัตโนมัติ หากยังไม่มีไฟล์ plant_health.db
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Plant Health Prediction & Management API")

# อนุญาตให้เรียก API ข้ามหน้าเว็บได้ (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# นำ API ทั้ง 2 กลุ่มที่เราเขียนไว้มาเชื่อมเข้ากับแอพหลัก และตั้งให้อยู่ภายใต้ /api
app.include_router(fields.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")

# ตั้งค่าโฟลเดอร์ static เพื่อให้ FastAPI รู้จักไฟล์ HTML/CSS/JS
static_path = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_path, exist_ok=True) # สร้างโฟลเดอร์กันเหนียวถ้ายังไม่มี
app.mount("/static", StaticFiles(directory=static_path), name="static")

# เมื่อเข้า URL หน้าแรก (http://localhost:8000/) ให้แสดงไฟล์ index.html
@app.get("/")
def serve_frontend():
    index_file = os.path.join(static_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Backend พร้อมใช้งานแล้ว! แต่ยังไม่พบไฟล์หน้าเว็บ /static/index.html"}