"""
Plant Health Prediction — Interactive Test Script
==================================================
รับ input ค่า Soil_Moisture และ Nitrogen_Level จากผู้ใช้
แล้วทำนายสถานะสุขภาพพืช (Healthy / Moderate Stress / High Stress)

ใช้ model: Decision Tree (best model จาก EXP2 — Feature Selection)
ใช้ features: Soil_Moisture, Nitrogen_Level
ใช้ scaler: StandardScaler (fitted on all 12 features)

Usage:
  python predict_health.py                         # โหมด interactive
  python predict_health.py --soil 25.0 --nitrogen 30.0   # โหมด command-line
"""

import argparse
import sys
import os
import joblib
import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# Configuration — ต้องตรงกับ notebook
# ──────────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("saved_models", "best_model_exp2_feat_sel_decision_tree.pkl")
SCALER_PATH = os.path.join("saved_models", "scaler.pkl")

# ลำดับ feature ทั้ง 12 ตัวที่ scaler ถูก fit (ตรงกับ notebook)
ALL_FEATURE_NAMES = [
    'Plant_ID', 'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature',
    'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level',
    'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content',
    'Electrochemical_Signal',
]

# features ที่ถูกเลือกจาก Mutual Information (MI > 0.05)
SELECTED_FEATURES = ['Soil_Moisture', 'Nitrogen_Level']
SELECTED_INDICES  = [ALL_FEATURE_NAMES.index(f) for f in SELECTED_FEATURES]

# label mapping (ตรงกับ custom_mapping ใน notebook)
LABEL_MAP = {0: 'Healthy 🌿', 1: 'Moderate Stress 🟡', 2: 'High Stress 🔴'}

# ช่วงข้อมูลจาก dataset (ใช้แสดง hint ตอน input)
FEATURE_RANGES = {
    'Soil_Moisture':  (10.0, 40.0),
    'Nitrogen_Level': (10.0, 50.0),
}


def load_model_and_scaler():
    """โหลด model และ scaler จากไฟล์ที่ saved ไว้"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ไม่พบไฟล์ model: {MODEL_PATH}")
        print("   กรุณา run notebook modeling.ipynb ก่อนเพื่อ train และ save model")
        sys.exit(1)
    if not os.path.exists(SCALER_PATH):
        print(f"❌ ไม่พบไฟล์ scaler: {SCALER_PATH}")
        print("   กรุณา run notebook modeling.ipynb ก่อนเพื่อ save scaler")
        sys.exit(1)

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict(model, scaler, soil_moisture: float, nitrogen_level: float) -> dict:
    """
    ทำนายสถานะสุขภาพพืช จากค่า Soil_Moisture และ Nitrogen_Level

    Parameters
    ----------
    model        : trained model object
    scaler       : fitted StandardScaler
    soil_moisture : float — ค่าความชื้นในดิน (ประมาณ 10–40)
    nitrogen_level: float — ระดับไนโตรเจน (ประมาณ 10–50)

    Returns
    -------
    dict with keys:
        'prediction'   : str  — label ที่ทำนายได้
        'prediction_id': int  — รหัส label (0, 1, 2)
        'probabilities': dict — ความน่าจะเป็นของแต่ละ class (ถ้า model รองรับ)
    """
    # สร้าง array 12 features ทั้งหมด (ใส่ 0 ในตำแหน่งที่ไม่ใช้)
    # เนื่องจาก scaler ถูก fit กับ 12 features
    full_input = np.zeros((1, len(ALL_FEATURE_NAMES)))
    full_input[0, ALL_FEATURE_NAMES.index('Soil_Moisture')]  = soil_moisture
    full_input[0, ALL_FEATURE_NAMES.index('Nitrogen_Level')] = nitrogen_level

    # scale ทั้ง array แล้วเลือกเฉพาะ selected features
    scaled_full = scaler.transform(full_input)
    scaled_selected = scaled_full[:, SELECTED_INDICES]

    # ทำนาย
    pred_id = int(model.predict(scaled_selected)[0])
    result = {
        'prediction':    LABEL_MAP.get(pred_id, f"Unknown ({pred_id})"),
        'prediction_id': pred_id,
    }

    # ถ้า model มี predict_proba → แสดงความน่าจะเป็นด้วย
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(scaled_selected)[0]
        result['probabilities'] = {LABEL_MAP[i]: round(float(p), 4)
                                   for i, p in enumerate(probs)}

    return result


def interactive_mode(model, scaler):
    """โหมดรับ input จากผู้ใช้แบบ interactive"""
    print("=" * 60)
    print("🌱  Plant Health Prediction — Interactive Mode")
    print("=" * 60)
    print()
    print("กรุณาป้อนค่า feature ที่ต้องการทดสอบ")
    print("(พิมพ์ 'q' เพื่อออก)\n")

    while True:
        try:
            # ── รับ input ─────────────────────────────────────────────
            lo, hi = FEATURE_RANGES['Soil_Moisture']
            raw = input(f"  Soil_Moisture   (ช่วง {lo}–{hi}): ").strip()
            if raw.lower() == 'q':
                break
            soil_moisture = float(raw)

            lo, hi = FEATURE_RANGES['Nitrogen_Level']
            raw = input(f"  Nitrogen_Level  (ช่วง {lo}–{hi}): ").strip()
            if raw.lower() == 'q':
                break
            nitrogen_level = float(raw)

            # ── ทำนาย ─────────────────────────────────────────────────
            result = predict(model, scaler, soil_moisture, nitrogen_level)

            print()
            print(f"  📊 ผลการทำนาย: {result['prediction']}")
            if 'probabilities' in result:
                print(f"     ความน่าจะเป็น:")
                for label, prob in result['probabilities'].items():
                    bar = '█' * int(prob * 30)
                    print(f"       {label:25s}  {prob:.2%}  {bar}")
            print("-" * 60)
            print()

        except ValueError:
            print("  ❌ กรุณาป้อนตัวเลขที่ถูกต้อง\n")
        except KeyboardInterrupt:
            print("\n\nออกจากโปรแกรม")
            break

    print("\n👋 ขอบคุณที่ใช้งาน!")


def main():
    parser = argparse.ArgumentParser(
        description="Plant Health Prediction — ทำนายสุขภาพพืชจาก Soil_Moisture & Nitrogen_Level",
    )
    parser.add_argument("--soil",     type=float, default=None,
                        help="ค่า Soil_Moisture (ประมาณ 10–40)")
    parser.add_argument("--nitrogen", type=float, default=None,
                        help="ค่า Nitrogen_Level (ประมาณ 10–50)")
    args = parser.parse_args()

    model, scaler = load_model_and_scaler()

    # ── ถ้ามี argument → ทำนายตรง ๆ แล้วแสดงผล ──────────────────
    if args.soil is not None and args.nitrogen is not None:
        result = predict(model, scaler, args.soil, args.nitrogen)
        print(f"Soil_Moisture  = {args.soil}")
        print(f"Nitrogen_Level = {args.nitrogen}")
        print(f"Prediction     = {result['prediction']}")
        if 'probabilities' in result:
            print(f"Probabilities  = {result['probabilities']}")
        return

    # ── ไม่มี argument → เข้าโหมด interactive ─────────────────────
    interactive_mode(model, scaler)


if __name__ == "__main__":
    main()
