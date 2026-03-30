import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
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
LABEL_MAP = {0: 'Healthy', 1: 'Moderate Stress', 2: 'High Stress'}

def main():
    print("🌱 Loading Model and Data...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Model or Data not found!")
        return

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATA_PATH)

    # ---------------------------------------------------------
    # 1. Preprocess Data
    # ---------------------------------------------------------
    # The models were trained on features without the target string
    # Assuming 'Health_Status' is the target column in the CSV
    X_raw = df[['Soil_Moisture', 'Nitrogen_Level']]
    
    # We need to scale them. The scaler was fitted on all 12 features.
    # To correctly scale, we need full 12 features array.
    full_input = np.zeros((len(df), len(ALL_FEATURE_NAMES)))
    
    # Copy what we have into the correct columns
    for col in ALL_FEATURE_NAMES:
        if col in df.columns:
             full_input[:, ALL_FEATURE_NAMES.index(col)] = df[col].values

    # Scale using the fitted scaler
    scaled_full = scaler.transform(full_input)
    
    # Extract only the selected features for the model
    X_scaled = scaled_full[:, SELECTED_INDICES]
    X_scaled_df = pd.DataFrame(X_scaled, columns=SELECTED_FEATURES)

    # ---------------------------------------------------------
    # 2. Feature Importances
    # ---------------------------------------------------------
    print("\n📊 1. Calculating Feature Importances...")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for feat, imp in zip(SELECTED_FEATURES, importances):
            print(f"  - {feat}: {imp:.4f}")
        
        plt.figure(figsize=(8, 4))
        sns.barplot(x=importances, y=SELECTED_FEATURES, palette='viridis')
        plt.title('Decision Tree Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('insight_feature_importance.png', dpi=300)
        print("  ✅ Saved plot to 'insight_feature_importance.png'")
    else:
        print("  Model does not have feature_importances_ attribute.")

    # ---------------------------------------------------------
    # 3. SHAP Values
    # ---------------------------------------------------------
    print("\n🔮 2. Calculating SHAP Values...")
    try:
        # For Decision Trees, TreeExplainer is fast and exact
        explainer = shap.TreeExplainer(model)
        
        # We compute SHAP values using the same scaled subset
        # We can take a sample if the dataset is too huge, but it's likely small
        sample_size = min(1000, len(X_scaled_df))
        X_sample = X_scaled_df.sample(sample_size, random_state=42)
        
        shap_values = explainer.shap_values(X_sample)
        
        # DecisionTreeClassifier for multiclass returns a 3D array: (nb_samples, nb_features, nb_classes)
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            num_classes = shap_values.shape[2]
            print(f"  Model has {num_classes} classes.")
            # Plot for Class 0 (Healthy)
            print("  Generating SHAP summary for 'Healthy' class (Class 0)...")
            plt.figure()
            shap.summary_plot(shap_values[:, :, 0], X_sample, show=False)
            plt.title("SHAP Summary: Healthy Class")
            plt.tight_layout()
            plt.savefig('insight_shap_summary_class0.png', dpi=300, bbox_inches='tight')
            
            # Plot for Class 2 (High Stress) if available
            if num_classes > 2:
                print("  Generating SHAP summary for 'High Stress' class (Class 2)...")
                plt.figure()
                shap.summary_plot(shap_values[:, :, 2], X_sample, show=False)
                plt.title("SHAP Summary: High Stress Class")
                plt.tight_layout()
                plt.savefig('insight_shap_summary_class2.png', dpi=300, bbox_inches='tight')
        elif isinstance(shap_values, list):
            print(f"  Model has {len(shap_values)} classes.")
            # Plot for Class 0 (Healthy)
            print("  Generating SHAP summary for 'Healthy' class (Class 0)...")
            plt.figure()
            shap.summary_plot(shap_values[0], X_sample, show=False)
            plt.title("SHAP Summary: Healthy Class")
            plt.tight_layout()
            plt.savefig('insight_shap_summary_class0.png', dpi=300, bbox_inches='tight')
            
            # Plot for Class 2 (High Stress)
            if len(shap_values) > 2:
                print("  Generating SHAP summary for 'High Stress' class (Class 2)...")
                plt.figure()
                shap.summary_plot(shap_values[2], X_sample, show=False)
                plt.title("SHAP Summary: High Stress Class")
                plt.tight_layout()
                plt.savefig('insight_shap_summary_class2.png', dpi=300, bbox_inches='tight')
        else:
            # For binary classification or regressor
            plt.figure()
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title("SHAP Summary")
            plt.tight_layout()
            plt.savefig('insight_shap_summary.png', dpi=300, bbox_inches='tight')
            
        print("  ✅ Saved SHAP summary plots.")
        
    except Exception as e:
        print(f"  ❌ Error calculating SHAP: {e}")

    print("\n✅ All Insight generation completed successfully!")

if __name__ == '__main__':
    main()
