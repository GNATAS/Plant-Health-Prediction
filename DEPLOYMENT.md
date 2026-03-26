# 🌿 Plant Health Prediction — Model Deployment Guide

คู่มือการนำ Model ไปใช้จริงหลังจาก train และ save เรียบร้อยแล้ว

---

## Prerequisites

```bash
pip install -r requirements.txt
```

ไฟล์ที่จำเป็น (ได้จากการ train ใน notebook):

```
saved_models/
├── best_model.pkl          # trained model
├── scaler.pkl              # StandardScaler ที่ fit บน training data
└── selected_features.pkl   # list ของ features ที่เลือกไว้
```

---

## 1. Predict แบบ Script เดี่ยว

สร้างไฟล์ `predict.py`:

```python
import joblib
import pandas as pd

# ── Load artifacts ─────────────────────────────────────────────
model     = joblib.load('saved_models/best_model.pkl')
scaler    = joblib.load('saved_models/scaler.pkl')
sel_feats = joblib.load('saved_models/selected_features.pkl')

LABEL_MAP = {0: 'Healthy', 1: 'Moderate Stress', 2: 'High Stress'}

def predict_health(sensor_data: dict) -> dict:
    """
    Parameters
    ----------
    sensor_data : dict
        ค่า sensor จากพืช 1 ต้น เช่น:
        {
            'Plant_ID': 3,
            'Soil_Moisture': 22.5,
            'Ambient_Temperature': 25.1,
            'Soil_Temperature': 20.3,
            'Humidity': 60.0,
            'Light_Intensity': 450.0,
            'Soil_pH': 6.2,
            'Nitrogen_Level': 28.5,
            'Phosphorus_Level': 35.0,
            'Potassium_Level': 30.1,
            'Chlorophyll_Content': 40.2,
            'Electrochemical_Signal': 0.85
        }

    Returns
    -------
    dict : { 'label': str, 'code': int, 'probabilities': dict }
    """
    df_in   = pd.DataFrame([sensor_data])
    X       = df_in[sel_feats]          # เลือกเฉพาะ selected features
    X_sc    = scaler.transform(X)       # scale ด้วย scaler เดิม
    code    = int(model.predict(X_sc)[0])
    label   = LABEL_MAP[code]

    result = {'label': label, 'code': code}

    # ถ้า model รองรับ predict_proba (ไม่ใช่ SVM default)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_sc)[0]
        result['probabilities'] = {
            LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)
        }

    return result


# ── ตัวอย่างใช้งาน ─────────────────────────────────────────────
if __name__ == '__main__':
    sample = {
        'Plant_ID': 3,
        'Soil_Moisture': 12.0,           # ต่ำมาก → น่าจะ Stress
        'Ambient_Temperature': 30.0,
        'Soil_Temperature': 28.0,
        'Humidity': 40.0,
        'Light_Intensity': 300.0,
        'Soil_pH': 6.0,
        'Nitrogen_Level': 8.0,
        'Phosphorus_Level': 20.0,
        'Potassium_Level': 15.0,
        'Chlorophyll_Content': 25.0,
        'Electrochemical_Signal': 0.5,
    }

    result = predict_health(sample)
    print(f"Plant Status : {result['label']}")
    print(f"Code         : {result['code']}")
    if 'probabilities' in result:
        print("Probabilities:")
        for cls, prob in result['probabilities'].items():
            bar = '█' * int(prob * 20)
            print(f"  {cls:20s} {prob:.2%}  {bar}")
```

รัน:
```bash
python predict.py
```

---

## 2. Predict แบบ Batch (หลายต้นพร้อมกัน)

```python
import joblib
import pandas as pd

model     = joblib.load('saved_models/best_model.pkl')
scaler    = joblib.load('saved_models/scaler.pkl')
sel_feats = joblib.load('saved_models/selected_features.pkl')
LABEL_MAP = {0: 'Healthy', 1: 'Moderate Stress', 2: 'High Stress'}

def predict_batch(df_new: pd.DataFrame) -> pd.DataFrame:
    """
    df_new : DataFrame ของข้อมูล sensor หลายแถว
    returns : df_new พร้อม column 'Predicted_Status' เพิ่มมา
    """
    X    = df_new[sel_feats]
    X_sc = scaler.transform(X)
    preds = model.predict(X_sc)
    df_out = df_new.copy()
    df_out['Predicted_Code']   = preds
    df_out['Predicted_Status'] = [LABEL_MAP[p] for p in preds]
    return df_out

# ใช้งาน
new_data   = pd.read_csv('new_sensor_readings.csv')
results_df = predict_batch(new_data)
results_df.to_csv('predictions_output.csv', index=False)
print(results_df[['Plant_ID', 'Predicted_Status']].to_string())
```

---

## 3. Serve เป็น REST API ด้วย FastAPI

```bash
pip install fastapi uvicorn
```

สร้าง `app.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

app       = FastAPI(title="Plant Health Prediction API")
model     = joblib.load('saved_models/best_model.pkl')
scaler    = joblib.load('saved_models/scaler.pkl')
sel_feats = joblib.load('saved_models/selected_features.pkl')
LABEL_MAP = {0: 'Healthy', 1: 'Moderate Stress', 2: 'High Stress'}

class SensorInput(BaseModel):
    Plant_ID: int
    Soil_Moisture: float
    Ambient_Temperature: float
    Soil_Temperature: float
    Humidity: float
    Light_Intensity: float
    Soil_pH: float
    Nitrogen_Level: float
    Phosphorus_Level: float
    Potassium_Level: float
    Chlorophyll_Content: float
    Electrochemical_Signal: float

class PredictionOutput(BaseModel):
    label: str
    code: int
    probabilities: dict | None = None

@app.get('/')
def root():
    return {'message': 'Plant Health Prediction API is running 🌿'}

@app.post('/predict', response_model=PredictionOutput)
def predict(data: SensorInput):
    df_in  = pd.DataFrame([data.model_dump()])
    X_sc   = scaler.transform(df_in[sel_feats])
    code   = int(model.predict(X_sc)[0])
    result = {'label': LABEL_MAP[code], 'code': code}
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_sc)[0]
        result['probabilities'] = {LABEL_MAP[i]: round(float(p), 4)
                                   for i, p in enumerate(proba)}
    return result
```

รัน server:
```bash
uvicorn app:app --reload --port 8000
```

เรียกใช้งาน:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"Plant_ID": 3, "Soil_Moisture": 12.0, "Ambient_Temperature": 30.0,
          "Soil_Temperature": 28.0, "Humidity": 40.0, "Light_Intensity": 300.0,
          "Soil_pH": 6.0, "Nitrogen_Level": 8.0, "Phosphorus_Level": 20.0,
          "Potassium_Level": 15.0, "Chlorophyll_Content": 25.0,
          "Electrochemical_Signal": 0.5}'
```

ผลลัพธ์:
```json
{
  "label": "High Stress",
  "code": 2,
  "probabilities": {
    "Healthy": 0.0312,
    "Moderate Stress": 0.1245,
    "High Stress": 0.8443
  }
}
```

Swagger UI อัตโนมัติ: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 4. Input Validation & Error Handling

```python
VALID_RANGES = {
    'Soil_Moisture':         (0, 100),
    'Ambient_Temperature':   (0, 50),
    'Soil_Temperature':      (0, 50),
    'Humidity':              (0, 100),
    'Light_Intensity':       (0, 2000),
    'Soil_pH':               (0, 14),
    'Nitrogen_Level':        (0, 100),
    'Phosphorus_Level':      (0, 100),
    'Potassium_Level':       (0, 100),
    'Chlorophyll_Content':   (0, 100),
    'Electrochemical_Signal': (0, 5),
}

def validate_input(data: dict) -> list[str]:
    errors = []
    for field, (lo, hi) in VALID_RANGES.items():
        val = data.get(field)
        if val is None:
            errors.append(f"Missing field: {field}")
        elif not (lo <= val <= hi):
            errors.append(f"{field}={val} out of range [{lo}, {hi}]")
    return errors

# ใช้งาน
errors = validate_input(sensor_data)
if errors:
    print("❌ Invalid input:", errors)
else:
    result = predict_health(sensor_data)
```

---

## 5. สิ่งที่ต้องระวังใน Production

| ประเด็น | คำแนะนำ |
|---------|---------|
| **Data Drift** | ตรวจสอบว่า distribution ของ sensor ยังใกล้เคียง training data |
| **Retraining** | ควร retrain เมื่อ F1 บน data ใหม่ลดลงเกิน 5% |
| **Model Version** | เก็บ version ของ model + วันที่ train ไว้ด้วย |
| **Scaler Sync** | scaler ต้องคู่กับ model เสมอ — ห้าม mix ข้าม version |
| **Logging** | Log ทุก prediction พร้อม timestamp และ input values |
| **SVM + Proba** | ถ้าใช้ SVM ต้อง `SVC(probability=True)` ถึงจะใช้ `predict_proba` ได้ |
