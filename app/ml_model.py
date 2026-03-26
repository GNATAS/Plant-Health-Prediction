import joblib
import pandas as pd
import os

# 1. ตั้งค่า Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_models', 'best_model_exp2_feat_sel_decision_tree.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'saved_models', 'scaler.pkl')
FEATS_PATH = os.path.join(BASE_DIR, 'saved_models', 'selected_features.pkl')

# พิมพ์ให้ดูใน Terminal ว่ามันไปหาไฟล์ที่ไหน
print(f"🔍 กำลังค้นหา Model ที่: {MODEL_PATH}")

# 2. โหลดโมเดล
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    sel_feats = joblib.load(FEATS_PATH)
    print("✅ โหลดโมเดล Machine Learning สำเร็จ!")
except Exception as e:
    print(f"❌ โหลดโมเดลไม่สำเร็จ! สาเหตุ: {e}")
    model, scaler, sel_feats = None, None, None

LABEL_MAP = {0: 'Healthy', 1: 'Moderate Stress', 2: 'High Stress'}

def predict_health(sensor_data: dict) -> dict:
    if model is None or scaler is None or sel_feats is None:
        raise ValueError("Model หรือไฟล์ Features ยังไม่ถูกโหลดเข้าสู่ระบบ")

    # 1. สร้าง Dictionary โดยใช้ชื่อคอลัมน์ "ตัวพิมพ์ใหญ่" ให้ตรงกับตอน Train Model เป๊ะๆ
    mapped_data = {
        'Plant_ID': sensor_data.get('field_id', 0),
        'Soil_Moisture': sensor_data.get('soil_moisture', 0.0),
        'Ambient_Temperature': sensor_data.get('ambient_temperature', 0.0),
        'Soil_Temperature': sensor_data.get('soil_temperature', 0.0),
        'Humidity': sensor_data.get('humidity', 0.0),
        'Light_Intensity': sensor_data.get('light_intensity', 0.0),
        'Soil_pH': sensor_data.get('soil_ph', 0.0),
        'Nitrogen_Level': sensor_data.get('nitrogen_level', 0.0),
        'Phosphorus_Level': sensor_data.get('phosphorus_level', 0.0),
        'Potassium_Level': sensor_data.get('potassium_level', 0.0),
        'Chlorophyll_Content': sensor_data.get('chlorophyll_content', 0.0),
        'Electrochemical_Signal': sensor_data.get('electrochemical_signal', 0.0)
    }

    # 2. แปลงเป็น DataFrame (ตอนนี้หัวตารางจะเป็นตัวพิมพ์ใหญ่แล้ว)
    df_in = pd.DataFrame([mapped_data])
    
    # 3. เรียงลำดับคอลัมน์ทั้ง 12 ตัวให้ตรงกับตอน Train Scaler (สำคัญมาก!)
    all_features = [
        'Plant_ID', 'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature', 
        'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level', 
        'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content', 
        'Electrochemical_Signal'
    ]
    
    # ดึงข้อมูลตามลำดับ 12 ตัวแปร
    X_all = df_in[all_features]
    
    # 4. Scale ข้อมูล (ต้องใช้ 12 ตัวตามที่ Scaler ถูกสอนมา)
    X_scaled_all = scaler.transform(X_all)
    
    # แปลงผลลัพธ์จากการ Scale กลับเป็น DataFrame เพื่อให้เลือกตามชื่อคอลัมน์ได้
    df_scaled = pd.DataFrame(X_scaled_all, columns=all_features)
    
# ... (ส่วนที่ 1-4 เหมือนเดิม) ...
    
    # 5. เลือกเฉพาะ 2 ตัวแปรที่โมเดล EXP2 นี้ต้องการจริงๆ
    # จากผลการรัน MI Score ของคุณ พบว่ามีแค่ 2 ตัวนี้ที่ถูกใช้เทรนโมเดล
    actual_features_used = ['Soil_Moisture', 'Nitrogen_Level']
    X_final = df_scaled[actual_features_used]
    
    # 6. ทำนายผล
  # ... (โค้ดส่วนบนคงเดิม) ...

    # 6. ทำนายผล (เติม .values เข้าไปทั้ง 2 จุด)
    try:
        # เติม .values เพื่อส่งเฉพาะค่าตัวเลขไปให้โมเดล
        code = int(model.predict(X_final.values)[0]) 
        
        label = LABEL_MAP.get(code, "Unknown")
        result = {'label': label, 'code': code}

        if hasattr(model, 'predict_proba'):
            # เติม .values ตรงนี้ด้วยครับ
            proba = model.predict_proba(X_final.values)[0] 
            result['probabilities'] = {
                LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)
            }
        return result
    except Exception as e:
        raise ValueError(f"เกิดข้อผิดพลาดขณะทำนาย: {str(e)}")