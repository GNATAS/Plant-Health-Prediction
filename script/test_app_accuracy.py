import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("  TESTING ACCURACY: CURRENT APP VALUES (HARDCODED)")
print("=" * 60)

# 1. Load data
df = pd.read_csv("data/plant_health_data.csv")
custom_mapping = {'High Stress': 2, 'Moderate Stress': 1, 'Healthy': 0}
df['label'] = df['Plant_Health_Status'].map(custom_mapping)
label_names = {0: 'Healthy', 1: 'Moderate Stress', 2: 'High Stress'}

# 2. Load model & scaler
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

# 3. Predict Single Function
def predict_one(soil_val, nitrogen_val):
    full_input = np.zeros((1, len(ALL_FEATURE_NAMES)))
    full_input[0, ALL_FEATURE_NAMES.index('Soil_Moisture')]  = soil_val
    full_input[0, ALL_FEATURE_NAMES.index('Nitrogen_Level')] = nitrogen_val
    scaled_full = scaler.transform(full_input)
    scaled_selected = scaled_full[:, SELECTED_INDICES]
    return int(model.predict(scaled_selected)[0])

# 4. Prepare Test Data
df_work = df.copy()
df_work = df_work.drop(['Timestamp', 'Plant_Health_Status', 'Plant_Name'], axis=1)
X = df_work.drop(columns=['label'])
y = df_work['label']
X_scaled = scaler.transform(X)
_, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

sm_idx = ALL_FEATURE_NAMES.index('Soil_Moisture')
nl_idx = ALL_FEATURE_NAMES.index('Nitrogen_Level')
X_test_unscaled = scaler.inverse_transform(X_test)

# 5. App mapping rules
def map_app_soil(raw_val):
    # Rule based on Actionable Insights in app.js
    if raw_val < 20: return 15
    elif raw_val >= 30: return 35
    else: return 25

def map_app_nitrogen(raw_val):
    # Rule based on Actionable Insights in app.js
    if raw_val <= 19.5: return 18
    elif raw_val >= 21: return 30
    else: return 20.5

# 6. Evaluate
y_pred_app = []
for i in range(len(X_test)):
    raw_sm = X_test_unscaled[i, sm_idx]
    raw_nl = X_test_unscaled[i, nl_idx]
    
    app_sm = map_app_soil(raw_sm)
    app_nl = map_app_nitrogen(raw_nl)
    
    pred = predict_one(app_sm, app_nl)
    y_pred_app.append(pred)

y_pred_app = np.array(y_pred_app)
acc_app = accuracy_score(y_test, y_pred_app)

# Exact Accuracy
X_test_fs = X_test[:, SELECTED_INDICES]
y_pred_direct = model.predict(X_test_fs)
acc_direct = accuracy_score(y_test, y_pred_direct)

print(f"  Soil mapping:     <20 -> 15 | >=30 -> 35 | else -> 25")
print(f"  Nitrogen mapping: <=19.5 -> 18 | >=21 -> 30 | else -> 20.5")
print(f"")
print(f"  Accuracy (exact values):     {acc_direct:.4f}")
print(f"  Accuracy (App current vals): {acc_app:.4f}")
print(f"  Accuracy drop:               {acc_direct - acc_app:.4f}")

from sklearn.metrics import classification_report
print("\n  Classification Report (App current vals):")
print(classification_report(y_test, y_pred_app, target_names=list(label_names.values())))
print("=" * 60)
