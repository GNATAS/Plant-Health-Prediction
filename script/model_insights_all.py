import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = os.path.join("saved_models", "exp1_baseline_decision_tree.pkl")
SCALER_PATH = os.path.join("saved_models", "scaler.pkl")
DATA_PATH = os.path.join("data", "plant_health_data.csv")

ALL_FEATURE_NAMES = [
    'Plant_ID', 'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature',
    'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level',
    'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content',
    'Electrochemical_Signal',
]

def main():
    print("🌱 Loading Model and Data (Exp 1: Baseline - All Features)...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print(f"Model ({MODEL_PATH}) or Data not found!")
        return

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATA_PATH)

    # Prepare features
    full_input = np.zeros((len(df), len(ALL_FEATURE_NAMES)))
    for col in ALL_FEATURE_NAMES:
        if col in df.columns:
             full_input[:, ALL_FEATURE_NAMES.index(col)] = df[col].values

    # Scale using the fitted scaler
    X_scaled = scaler.transform(full_input)
    X_scaled_df = pd.DataFrame(X_scaled, columns=ALL_FEATURE_NAMES)

    # 1. Feature Importances
    print("\n📊 1. Calculating Feature Importances...")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        sorted_features = [ALL_FEATURE_NAMES[i] for i in indices]
        sorted_importances = importances[indices]
        
        print("Feature Importances:")
        for feat, imp in zip(sorted_features, sorted_importances):
            print(f"  - {feat}: {imp:.4f}")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
        plt.title('Baseline Decision Tree Feature Importances (All Features)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('insight_all_feature_importance.png', dpi=300)
        print("  ✅ Saved plot to 'insight_all_feature_importance.png'")
    else:
        print("  Model does not have feature_importances_ attribute.")

    # 2. SHAP Values
    print("\n🔮 2. Calculating SHAP Values...")
    try:
        explainer = shap.TreeExplainer(model)
        sample_size = min(1000, len(X_scaled_df))
        X_sample = X_scaled_df.sample(sample_size, random_state=42)
        
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            num_classes = shap_values.shape[2]
            print(f"  Model has {num_classes} classes.")
            # Plot for Class 0 (Healthy)
            print("  Generating SHAP summary for 'Healthy' class (Class 0)...")
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values[:, :, 0], X_sample, show=False)
            plt.title("SHAP Summary: Healthy Class (All Features)")
            plt.tight_layout()
            plt.savefig('insight_all_shap_summary_class0.png', dpi=300, bbox_inches='tight')
            
            # Plot for Class 2 (High Stress) if available
            if num_classes > 2:
                print("  Generating SHAP summary for 'High Stress' class (Class 2)...")
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values[:, :, 2], X_sample, show=False)
                plt.title("SHAP Summary: High Stress Class (All Features)")
                plt.tight_layout()
                plt.savefig('insight_all_shap_summary_class2.png', dpi=300, bbox_inches='tight')
        elif isinstance(shap_values, list): # For older SHAP versions or certain models
            print(f"  Model has {len(shap_values)} classes.")
            print("  Generating SHAP summary for 'Healthy' class (Class 0)...")
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values[0], X_sample, show=False)
            plt.title("SHAP Summary: Healthy Class (All Features)")
            plt.tight_layout()
            plt.savefig('insight_all_shap_summary_class0.png', dpi=300, bbox_inches='tight')
            
            if len(shap_values) > 2:
                print("  Generating SHAP summary for 'High Stress' class (Class 2)...")
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values[2], X_sample, show=False)
                plt.title("SHAP Summary: High Stress Class (All Features)")
                plt.tight_layout()
                plt.savefig('insight_all_shap_summary_class2.png', dpi=300, bbox_inches='tight')
        else:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title("SHAP Summary (All Features)")
            plt.tight_layout()
            plt.savefig('insight_all_shap_summary.png', dpi=300, bbox_inches='tight')
        
        print("  ✅ Saved SHAP summary plots.")
        
    except Exception as e:
        print(f"  ❌ Error calculating SHAP: {e}")

    print("\n✅ All Insight generation completed successfully!")

if __name__ == '__main__':
    main()
