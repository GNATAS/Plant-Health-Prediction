from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/fields", tags=["Fields Management"])

@router.post("/", response_model=schemas.FieldResponse)
def create_field(field: schemas.FieldCreate, db: Session = Depends(get_db)):
    """API สำหรับสร้างแปลงปลูกใหม่"""
    db_field = models.Field(**field.model_dump())
    db.add(db_field)
    db.commit()
    db.refresh(db_field)
    return db_field

@router.get("/", response_model=List[schemas.FieldResponse])
def read_fields(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """API สำหรับดึงรายชื่อแปลงปลูกทั้งหมด"""
    fields = db.query(models.Field).offset(skip).limit(limit).all()
    return fields

@router.get("/{field_id}", response_model=schemas.FieldWithHistoryResponse)
def read_field_with_history(field_id: int, db: Session = Depends(get_db)):
    """API สำหรับดูข้อมูลแปลง 1 แปลง พร้อมประวัติการทำนายผลทั้งหมดที่ผ่านมา"""
    field = db.query(models.Field).filter(models.Field.id == field_id).first()
    if field is None:
        raise HTTPException(status_code=404, detail="ไม่พบข้อมูลแปลงนี้")
    return field