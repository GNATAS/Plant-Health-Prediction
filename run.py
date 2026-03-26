import uvicorn

if __name__ == "__main__":
    # สั่งรันแอป FastAPI ที่อยู่ในโฟลเดอร์ app ไฟล์ main.py
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)