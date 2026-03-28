from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

# --- Schemas สำหรับ "แปลงปลูก" (Field) ---
class FieldBase(BaseModel):
    name: str
    plant_type: str

class FieldCreate(FieldBase):
    pass

class FieldResponse(FieldBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# --- Schemas สำหรับ "การทำนาย" (Prediction) ---
class PredictionInput(BaseModel):
    field_id: int
    soil_moisture: float
    ambient_temperature: float
    soil_temperature: float
    humidity: float
    light_intensity: float
    soil_ph: float
    nitrogen_level: float
    phosphorus_level: float
    potassium_level: float
    chlorophyll_content: float
    electrochemical_signal: float

class PredictionResponse(BaseModel):
    id: int
    field_id: int
    predicted_status: str
    predicted_code: int
    probabilities: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Schema สำหรับเวลาดึงข้อมูลแปลง พร้อมประวัติของแปลงนั้น
class FieldWithHistoryResponse(FieldResponse):
    predictions: List[PredictionResponse] = []