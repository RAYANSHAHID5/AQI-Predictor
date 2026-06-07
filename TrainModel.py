import pandas as pd
import numpy as np
import hopsworks
import joblib
import os
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_and_select_champion():
    # 1. Connect to Hopsworks and fetch data
    print(" Connecting to Hopsworks Feature Store...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    try:
        feature_group = fs.get_feature_group(name="lahore_aqi_features", version=2)
        df = feature_group.read()
    except Exception as e:
        print(f" Failed to fetch data: {e}")
        return

    # Sort chronologically to prevent temporal data leakage
    df = df.sort_values("timestamp")

    # 2. Prepare Data
    X = df.drop(columns=["timestamp", "target_aqi"], errors='ignore')
    y = df["target_aqi"]

    # 80/20 Chronological Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 3. Define the Competitors (Statistical, ML, Deep Learning)
    models = {
        "Statistical (Ridge)": Ridge(alpha=1.0),
        "Machine Learning (XGBoost)": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        "Deep Learning (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }

    print("\n Initiating Automated Model Tournament...")
    best_model = None
    best_name = ""
    lowest_rmse = float('inf')
    best_r2 = 0

    # 4. Train and Evaluate All Models
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        print(f"   ↳ RMSE: {rmse:.2f} | R²: {r2:.2f}")

        # Runtime Champion Selection
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best_r2 = r2
            best_model = model
            best_name = name

    print(f"\n CHAMPION CROWNED: {best_name} (RMSE: {lowest_rmse:.2f})")

    # 5. Generate SHAP Explanations for the Champion
    print(f" Generating Explainable AI (SHAP) for {best_name}...")
    plt.figure()
    try:
        if "XGBoost" in best_name:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)
        else:
            # Use KernelExplainer for non-tree models (Ridge, MLP)
            explainer = shap.KernelExplainer(best_model.predict, shap.sample(X_train, 100))
            shap_values = explainer.shap_values(X_test)

        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig("shap_summary.png", bbox_inches='tight')
        print(" SHAP summary plot saved.")
    except Exception as e:
        print(f" SHAP generation encountered an issue: {e}")

    # 6. Save and Register the Champion to Hopsworks
    print(" Uploading Champion to Hopsworks Model Registry...")
    joblib.dump(best_model, 'aqi_model.pkl')

    mr = project.get_model_registry()
    model_dir = "lahore_aqi_model_dir"
    os.makedirs(model_dir, exist_ok=True)
    os.system(f"mv aqi_model.pkl {model_dir}/")

    aqi_model = mr.python.create_model(
        name="lahore_aqi_model",
        metrics={"RMSE": lowest_rmse, "R2": best_r2, "Algorithm": best_name},
        description=f"Automated champion model. Winner: {best_name}"
    )
    aqi_model.save(model_dir)
    print("Champion successfully deployed to production!")


if __name__ == "__main__":
    train_and_select_champion()