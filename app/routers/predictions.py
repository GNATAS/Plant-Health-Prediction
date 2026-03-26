from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json

from .. import models, schemas
from ..database import get_db
from ..ml_model import predict_health  # ดึงฟังก์ชันทำนายผลมาจากไฟล์ ml_model.py ของคุณ

router = APIRouter(prefix="/predictions", tags=["ML Predictions"])

@router.post("/", response_model=schemas.PredictionResponse)
def make_prediction(prediction_in: schemas.PredictionInput, db: Session = Depends(get_db)):
    """API สำหรับรับค่า Sensor ไปทำนายผลและบันทึกประวัติ"""
    
    # 1. เช็คก่อนว่าแปลง (Field ID) นี้มีอยู่จริงไหมใน Database
    field = db.query(models.Field).filter(models.Field.id == prediction_in.field_id).first()
    if not field:
        raise HTTPException(status_code=404, detail="ไม่พบรหัสแปลง (Field ID) นี้ในระบบ")

    # 2. แปลงข้อมูล Input ให้เป็น Dictionary เพื่อโยนเข้า ML Model
    sensor_data = prediction_in.model_dump()
    
    # 3. ให้ ML Model ทำงาน (ถ้ามี Error ให้แจ้งเตือน)
    try:
        # สมมติว่าใน ml_model.py ของคุณ ฟังก์ชัน predict_health รับ dict แล้วคืนค่า {'label': ..., 'code': ..., 'probabilities': ...}
        ml_result = predict_health(sensor_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการรันโมเดล: {str(e)}")

    # 4. นำผลลัพธ์และค่า Sensor มาเตรียมบันทึกลง Database
    db_prediction = models.PredictionHistory(
        field_id=prediction_in.field_id,
        soil_moisture=prediction_in.soil_moisture,
        ambient_temperature=prediction_in.ambient_temperature,
        soil_temperature=prediction_in.soil_temperature,
        humidity=prediction_in.humidity,
        light_intensity=prediction_in.light_intensity,
        soil_ph=prediction_in.soil_ph,
        nitrogen_level=prediction_in.nitrogen_level,
        phosphorus_level=prediction_in.phosphorus_level,
        potassium_level=prediction_in.potassium_level,
        chlorophyll_content=prediction_in.chlorophyll_content,
        electrochemical_signal=prediction_in.electrochemical_signal,
        
        # ใส่ผลลัพธ์จาก Model
        predicted_status=ml_result['label'],
        predicted_code=ml_result['code'],
        probabilities=json.dumps(ml_result.get('probabilities', {}))
    )

    # 5. บันทึกลง Database
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    
    return db_prediction