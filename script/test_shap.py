import os
import joblib
import pandas as pd
import numpy as np
import shap

MODEL_PATH = os.path.join("saved_models", "best_model_exp2_feat_sel_decision_tree.pkl")
SCALER_PATH = os.path.join("saved_models", "scaler.pkl")
DATA_PATH = os.path.join("data", "plant_health_data.csv")

ALL_FEATURE_NAMES = [
    'Plant_ID', 'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature',
    'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level',
    'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content',
    'Electrochemical_Signal',
]
SELECTED_FEATURES = ['Soil_Moisture', 'Nitrogen_Level']
SELECTED_INDICES = [ALL_FEATURE_NAMES.index(f) for f in SELECTED_FEATURES]

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(DATA_PATH)

X_raw = df[['Soil_Moisture', 'Nitrogen_Level']]
full_input = np.zeros((len(df), len(ALL_FEATURE_NAMES)))
for col in ALL_FEATURE_NAMES:
    if col in df.columns:
         full_input[:, ALL_FEATURE_NAMES.index(col)] = df[col].values
scaled_full = scaler.transform(full_input)
X_scaled = scaled_full[:, SELECTED_INDICES]
X_scaled_df = pd.DataFrame(X_scaled, columns=SELECTED_FEATURES)

explainer = shap.TreeExplainer(model)
X_sample = X_scaled_df.sample(10, random_state=42)

# test what shap_values returns
out_array = explainer.shap_values(X_sample)
print("Type of shap_values():", type(out_array))
if isinstance(out_array, list):
    print("List length:", len(out_array))
    print("Shape of first array:", out_array[0].shape)
elif isinstance(out_array, np.ndarray):
    print("Shape of array:", out_array.shape)

out_obj = explainer(X_sample)
print("Type of explainer():", type(out_obj))
if hasattr(out_obj, 'shape'):
    print("Shape of Explanation object:", out_obj.shape)
