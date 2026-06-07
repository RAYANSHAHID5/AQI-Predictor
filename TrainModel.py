"""
pulls historical features from the feature store, trains and compares a few
different models (Ridge Regression, Random Forest, and XGBoost) then evaluates
them using RMSE, MAE, and R².
also generates SHAP explanations and registers the best-performing model.
"""


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

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT_NAME = "AQI_Predictor_RS"

def benchmark_and_register_model():
    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(
        project=PROJECT_NAME,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()

    # Fetch historical features and targets from Feature Store
    aqi_fg = fs.get_feature_group(name="lahore_aqi_features", version=2)
    df = aqi_fg.read()

    # Sequential split required for time series forecasting to prevent data leakage
    df = df.sort_values('timestamp')
    X = df.drop(columns=['timestamp', 'target_aqi'])
    y = df['target_aqi']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Experiment with various ML models per requirements
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

    # Evaluate performance using RMSE, MAE, and R² metrics
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

    # Save benchmark reporting
    results_df = pd.DataFrame(results)
    with open("benchmark_report.txt", "w") as f:
        f.write("AQI PREDICTOR - MODEL BENCHMARKING REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(results_df.to_string(index=False) + "\n")
        f.write("=" * 50 + "\n")
        f.write(f"WINNING MODEL SELECTED: {best_model_name}\n")

    # Advanced Analytics: Use SHAP for feature importance explanations
    print("\nGenerating SHAP Explainer Plot...")
    try:
        if best_model_name in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(best_model_instance)
            shap_values = explainer.shap_values(X_test)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            plt.savefig("shap_summary.png", bbox_inches="tight", dpi=300)
            plt.close()
        else:
            print(f"SHAP Explainer bypassed for linear model methodology.")
    except Exception as e:
        print(f"Could not generate SHAP plot: {e}")

    # Store trained model in Model Registry
    model_filename = "aqi_model.pkl"
    joblib.dump(best_model_instance, model_filename)

    mr = project.get_model_registry()
    hopsworks_model = mr.python.create_model(
        name="lahore_aqi_model",
        metrics=best_metrics,
        description=f"Champion {best_model_name} model trained on real DHA Lahore data."
    )
    hopsworks_model.save(model_filename)

    if os.path.exists(model_filename):
        os.remove(model_filename)

if __name__ == "__main__":
    benchmark_and_register_model()