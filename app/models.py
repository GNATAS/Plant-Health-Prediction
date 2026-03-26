from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Field(Base):
    __tablename__ = "fields"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)           # ชื่อแปลง เช่น "แปลง A"
    plant_type = Column(String)                 # ชนิดพืช เช่น "Spinach"
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # ความสัมพันธ์: 1 แปลง มีได้หลายประวัติการทำนาย
    predictions = relationship("PredictionHistory", back_populates="field", cascade="all, delete-orphan")

class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    field_id = Column(Integer, ForeignKey("fields.id"))
    
    # เก็บค่า Sensor (อ้างอิงจากตอนที่คุณ Train Model)
    soil_moisture = Column(Float)
    ambient_temperature = Column(Float)
    soil_temperature = Column(Float)
    humidity = Column(Float)
    light_intensity = Column(Float)
    soil_ph = Column(Float)
    nitrogen_level = Column(Float)
    phosphorus_level = Column(Float)
    potassium_level = Column(Float)
    chlorophyll_content = Column(Float)
    electrochemical_signal = Column(Float)
    
    # เก็บผลลัพธ์การพยากรณ์
    predicted_status = Column(String)           # เช่น "High Stress"
    predicted_code = Column(Integer)            # เช่น 0, 1, 2
    probabilities = Column(String, nullable=True) # เก็บเป็น JSON String เผื่อไว้แสดงผล %
    created_at = Column(DateTime, default=datetime.utcnow)

    field = relationship("Field", back_populates="predictions")