import hopsworks
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

# Load secret variables from the .env file locally
load_dotenv()

# --- Configuration ---
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT_NAME = "AQI_Predictor_RS"


def benchmark_and_register_model():
    print("🚀 Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(
        project=PROJECT_NAME,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()

    print("📥 Downloading training features...")
    aqi_fg = fs.get_feature_group(name="lahore_aqi_features", version=2)
    df = aqi_fg.read()

    print("🛠️ Preparing features and targets...")
    # CRITICAL FIX: Sort by time to prevent Data Leakage!
    df = df.sort_values('timestamp')

    X = df.drop(columns=['timestamp', 'target_aqi'])
    y = df['target_aqi']

    # CRITICAL FIX: Sequential split, not random (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Define models with proper scaling for linear algorithms
    models = {
        "Ridge Regression": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    }

    results = []
    best_model_name = None
    best_r2 = -float('inf')
    best_model_instance = None
    best_metrics = {}

    print("\n⚔️ Benchmarking candidate models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R² Score": f"{round(r2 * 100, 2)}%"
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_instance = model
            best_metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}

    # --- SAVE THE BENCHMARK REPORT ---
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("🏆 MODEL LEADERBOARD COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    with open("benchmark_report.txt", "w") as f:
        f.write("AQI PREDICTOR - MODEL BENCHMARKING REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(results_df.to_string(index=False) + "\n")
        f.write("=" * 50 + "\n")
        f.write(f"WINNING MODEL SELECTED: {best_model_name}\n")
    print("\n📝 SAVED: 'benchmark_report.txt' generated in your project folder.")

    print(f"\n🥇 Winning Model determined: {best_model_name} with R²: {best_metrics['r2'] * 100:.2f}%")

    # --- NEW REQUIREMENT: SHAP Feature Importance ---
    print("\n📊 Generating SHAP Explainer Plot...")
    try:
        if best_model_name in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(best_model_instance)
            shap_values = explainer.shap_values(X_test)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            plt.savefig("shap_summary.png", bbox_inches="tight", dpi=300)
            plt.close()
            print("✅ Saved SHAP plot to 'shap_summary.png'")
        else:
            print(f"⚠️ SHAP TreeExplainer skipped (best model is {best_model_name}, which is not a tree).")
    except Exception as e:
        print(f"⚠️ Could not generate SHAP plot: {e}")

    # --- Register the Champion ---
    model_filename = "aqi_model.pkl"
    print(f"\n💾 Saving {best_model_name} locally to {model_filename}...")
    joblib.dump(best_model_instance, model_filename)

    print("☁️ Connecting to the Hopsworks Model Registry...")
    mr = project.get_model_registry()

    hopsworks_model = mr.python.create_model(
        name="lahore_aqi_model",
        metrics=best_metrics,
        description=f"Champion {best_model_name} model trained on real DHA Lahore data."
    )

    print("🚀 Pushing model file to cloud registry...")
    hopsworks_model.save(model_filename)
    print("🎉 Success! The model is now safely stored in your cloud registry.")

    if os.path.exists(model_filename):
        os.remove(model_filename)


if __name__ == "__main__":
    benchmark_and_register_model()