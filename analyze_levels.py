"""
Analyze data to find threshold values for each level
for Soil_Moisture and Nitrogen_Level
and test accuracy of using level-based representative values vs raw values
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys
import io
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# -- Load data -------------------------------------------------------
df = pd.read_csv("data/plant_health_data.csv")
custom_mapping = {'High Stress': 2, 'Moderate Stress': 1, 'Healthy': 0}
df['label'] = df['Plant_Health_Status'].map(custom_mapping)
label_names = {0: 'Healthy', 1: 'Moderate Stress', 2: 'High Stress'}

print("=" * 70)
print("  Analysis: Soil_Moisture & Nitrogen_Level Distribution by Class")
print("=" * 70)

# -- 1. Stats per class -----------------------------------------------
for feat in ['Soil_Moisture', 'Nitrogen_Level']:
    print(f"\n{'-'*60}")
    print(f"  Feature: {feat}")
    print(f"{'-'*60}")
    for label_id in sorted(label_names.keys()):
        subset = df[df['label'] == label_id][feat]
        print(f"  {label_names[label_id]:20s}  "
              f"mean={subset.mean():.2f}  std={subset.std():.2f}  "
              f"min={subset.min():.2f}  max={subset.max():.2f}  "
              f"Q25={subset.quantile(0.25):.2f}  Q75={subset.quantile(0.75):.2f}")

# -- 2. Compute percentile thresholds ---------------------------------
sm_p33 = df['Soil_Moisture'].quantile(0.33)
sm_p67 = df['Soil_Moisture'].quantile(0.67)
nl_p33 = df['Nitrogen_Level'].quantile(0.33)
nl_p67 = df['Nitrogen_Level'].quantile(0.67)

print(f"\n{'='*70}")
print("  Percentile-based thresholds (P33/P67)")
print(f"{'='*70}")
print(f"  Soil_Moisture:   P33={sm_p33:.2f}  P67={sm_p67:.2f}")
print(f"  Nitrogen_Level:  P33={nl_p33:.2f}  P67={nl_p67:.2f}")

# -- 3. Representative values (median of each group) ------------------
sm_low_median  = df[df['Soil_Moisture'] <= sm_p33]['Soil_Moisture'].median()
sm_mid_median  = df[(df['Soil_Moisture'] > sm_p33) & (df['Soil_Moisture'] <= sm_p67)]['Soil_Moisture'].median()
sm_high_median = df[df['Soil_Moisture'] > sm_p67]['Soil_Moisture'].median()

nl_low_median  = df[df['Nitrogen_Level'] <= nl_p33]['Nitrogen_Level'].median()
nl_mid_median  = df[(df['Nitrogen_Level'] > nl_p33) & (df['Nitrogen_Level'] <= nl_p67)]['Nitrogen_Level'].median()
nl_high_median = df[df['Nitrogen_Level'] > nl_p67]['Nitrogen_Level'].median()

print(f"\n{'='*70}")
print("  Representative values for each level")
print(f"{'='*70}")

print(f"\n  Soil_Moisture (Dry/Moist/Wet):")
print(f"    Dry   (<=P33={sm_p33:.2f})   representative = {sm_low_median:.2f}")
print(f"    Moist (P33-P67)              representative = {sm_mid_median:.2f}")
print(f"    Wet   (>P67={sm_p67:.2f})    representative = {sm_high_median:.2f}")

print(f"\n  Nitrogen_Level (leaf color -- Yellow/Green/DarkGreen):")
print(f"    Yellow/Pale (<=P33={nl_p33:.2f})  representative = {nl_low_median:.2f}")
print(f"    Normal Green (P33-P67)            representative = {nl_mid_median:.2f}")
print(f"    Dark Green  (>P67={nl_p67:.2f})   representative = {nl_high_median:.2f}")

# -- 4. Load model & test accuracy ------------------------------------
model  = joblib.load("saved_models/best_model_exp2_feat_sel_decision_tree.pkl")
scaler = joblib.load("saved_models/scaler.pkl")

ALL_FEATURE_NAMES = [
    'Plant_ID', 'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature',
    'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level',
    'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content',
    'Electrochemical_Signal',
]
SELECTED_FEATURES = ['Soil_Moisture', 'Nitrogen_Level']
SELECTED_INDICES  = [ALL_FEATURE_NAMES.index(f) for f in SELECTED_FEATURES]

def predict_one(soil_val, nitrogen_val):
    full_input = np.zeros((1, len(ALL_FEATURE_NAMES)))
    full_input[0, ALL_FEATURE_NAMES.index('Soil_Moisture')]  = soil_val
    full_input[0, ALL_FEATURE_NAMES.index('Nitrogen_Level')] = nitrogen_val
    scaled_full = scaler.transform(full_input)
    scaled_selected = scaled_full[:, SELECTED_INDICES]
    return int(model.predict(scaled_selected)[0])

# -- 5. Test all 3x3 combinations ------------------------------------
print(f"\n{'='*70}")
print("  Prediction for all 3x3 level combinations")
print(f"{'='*70}")

soil_labels = ['Dry', 'Moist', 'Wet']
soil_vals   = [sm_low_median, sm_mid_median, sm_high_median]
nit_labels  = ['Yellow/Pale', 'Normal Green', 'Dark Green']
nit_vals    = [nl_low_median, nl_mid_median, nl_high_median]

print(f"\n  {'Soil':>12s}  {'Nitrogen (leaf color)':>20s}  {'SM val':>8s}  {'NL val':>8s}  -->  Prediction")
print(f"  {'-'*12}  {'-'*20}  {'-'*8}  {'-'*8}  ---  {'-'*20}")

for sl, sv in zip(soil_labels, soil_vals):
    for nl, nv in zip(nit_labels, nit_vals):
        pred_id = predict_one(sv, nv)
        pred_name = label_names[pred_id]
        print(f"  {sl:>12s}  {nl:>20s}  {sv:8.2f}  {nv:8.2f}  -->  {pred_name}")

# -- 6. Test set accuracy comparison ----------------------------------
print(f"\n{'='*70}")
print("  Test set accuracy comparison (240 samples)")
print(f"{'='*70}")

df_work = df.copy()
df_work = df_work.drop(['Timestamp', 'Plant_Health_Status', 'Plant_Name'], axis=1)
X = df_work.drop(columns=['label'])
y = df_work['label']
X_scaled = scaler.transform(X)
_, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Direct prediction with exact values
X_test_fs = X_test[:, SELECTED_INDICES]
y_pred_direct = model.predict(X_test_fs)
acc_direct = accuracy_score(y_test, y_pred_direct)

# Level-based prediction
def classify_to_level(val, t_low, t_high):
    if val <= t_low:
        return 0
    elif val <= t_high:
        return 1
    else:
        return 2

sm_representative = {0: sm_low_median, 1: sm_mid_median, 2: sm_high_median}
nl_representative = {0: nl_low_median, 1: nl_mid_median, 2: nl_high_median}

sm_idx = ALL_FEATURE_NAMES.index('Soil_Moisture')
nl_idx = ALL_FEATURE_NAMES.index('Nitrogen_Level')
X_test_unscaled = scaler.inverse_transform(X_test)

y_pred_level = []
for i in range(len(X_test)):
    raw_sm = X_test_unscaled[i, sm_idx]
    raw_nl = X_test_unscaled[i, nl_idx]
    
    sm_level = classify_to_level(raw_sm, sm_p33, sm_p67)
    nl_level = classify_to_level(raw_nl, nl_p33, nl_p67)
    
    pred = predict_one(sm_representative[sm_level], nl_representative[nl_level])
    y_pred_level.append(pred)

y_pred_level = np.array(y_pred_level)
acc_level = accuracy_score(y_test, y_pred_level)

print(f"\n  Accuracy (exact values):     {acc_direct:.4f}")
print(f"  Accuracy (level-based):      {acc_level:.4f}")
print(f"  Accuracy drop:               {acc_direct - acc_level:.4f}")

print(f"\n  Classification Report (level-based):")
print(classification_report(y_test, y_pred_level, target_names=list(label_names.values())))

# -- Summary ----------------------------------------------------------
print(f"\n{'='*70}")
print("  SUMMARY: Recommended representative values")
print(f"{'='*70}")
print(f"""
  Soil_Moisture:
    1. Dry         -> {sm_low_median:.2f}
    2. Moist       -> {sm_mid_median:.2f}
    3. Wet         -> {sm_high_median:.2f}

  Nitrogen_Level (leaf color):
    1. Yellow/Pale -> {nl_low_median:.2f}
    2. Normal Green-> {nl_mid_median:.2f}
    3. Dark Green  -> {nl_high_median:.2f}

  Accuracy with exact values:  {acc_direct:.4f}
  Accuracy with levels:        {acc_level:.4f}
""")
