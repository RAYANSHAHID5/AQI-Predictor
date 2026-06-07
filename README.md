# AQI Predictor

**Location:** DHA Phase 8, Lahore, Pakistan

**Developed by:** Rayan Shahid

This project is a serverless machine learning pipeline that predicts the Air Quality Index (AQI) for the next hour in Lahore. We built this to automate the entire workflow: from fetching live data to training multiple models and deploying the most accurate one to a web dashboard.

## Overview

Instead of relying on static models or manual data updates, this system runs entirely on its own. GitHub Actions handles the hourly data collection and the daily model retraining. During the retraining phase, the system tests three different model architectures, picks the winner, and uploads it to Hopsworks, which acts as our cloud-based feature store and model registry.

## Features

* **Automated Data Pipeline:** Fetches live weather and pollution data every hour and pushes it to the cloud.
* **Dynamic Model Selection:** Trains a Statistical (Ridge), Deep Learning (MLP), and Machine Learning (XGBoost) model every day. It evaluates their performance and deploys the one with the lowest error.
* **Serverless Architecture:** Relies on GitHub Actions and Hopsworks, eliminating the need for a dedicated, always-on backend server.
* **Custom Feature Engineering:** Calculates momentum variables, like the rate of AQI change from the previous hour, to help the model catch sudden pollution spikes.
* **Model Interpretability:** Automatically generates a SHAP summary plot during training so users can see exactly which variables are driving the predictions.
* **Live Dashboard:** A Streamlit web app that displays current telemetry, historical data trends, and the live forecast.

## Data Sources

| Source | Data Provided |
| --- | --- |
| **AQICN API** | Live pollution data and current AQI. |
| **Open-Meteo API** | 1-month historical backfill (approx. 670 hourly rows) for weather data. |
| **Hopsworks** | Cloud Feature Store for both historical and live data vectors. |

## Training Features

The models are trained on the following inputs:

* **Meteorological:** Temperature (°C), Relative Humidity (%), Wind Speed (km/h)
* **Pollutants:** PM2.5 (µg/m³), PM10 (µg/m³)
* **Time Variables:** Hour, Day, Month, Day of Week (to capture rush hour and weekly cycles)
* **Engineered:** `aqi_change_rate` (The current AQI minus the previous hour's AQI)

## System Architecture

```text
┌─────────────────┐     ┌─────────────────┐
│   AQICN API     │     │ Open-Meteo API  │
│ (Live Air Data) │     │ (Weather Data)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │   GitHub Actions      │
         │  (Hourly Collection)  │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Hopsworks (Cloud)    │
         │   Feature Store &     │
         │   Model Registry      │
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         │ Automated Tournament  │
         │  (Daily Retraining)   │
         └───────────┬───────────┘
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │  Ridge  │  │   MLP   │  │ XGBoost │
   │ (Stats) │  │(Deep L.)│  │  (ML)   │
   └─────────┘  └─────────┘  └─────────┘
        │            │            │
        └────────────┼────────────┘
                     ▼
         ┌───────────────────────┐
         │     Streamlit App     │
         │   (Live Dashboard)    │
         └───────────────────────┘

```

## GitHub Actions Workflows

**1. Hourly Data Pipeline (`feature_pipeline.yml`)**

* Runs every hour.
* Triggers `FeaturePipeline.py` to fetch the latest Lahore data, calculate the AQI momentum, format the data types, and upload the new row to the Hopsworks Feature Store.

**2. Daily Model Retraining (`training_pipeline.yml`)**

* Runs daily.
* Triggers `TrainModel.py` to download the latest dataset, perform a chronological train/test split, and run the benchmarking tournament between the Ridge, MLP, and XGBoost models.
* Generates new SHAP analytics and registers the winning artifact to the Hopsworks Model Registry.

## Project Structure

```text
AQI-Predictor/
├── app.py                          # Streamlit dashboard application
├── requirements.txt                # Python dependencies (Python 3.11 compatible)
├── FeaturePipeline.py              # Hourly data fetching & Hopsworks upload script
├── TrainModel.py                   # Automated Tournament, Benchmarking & SHAP generation
├── BackFillOpenMeteo.py            # Historical data collection (1-month backfill)
├── shap_summary.png                # Auto-generated model interpretability plot
└── .github/workflows/
    ├── hourly_features.yml        # Hourly data collection workflow
    └── daily_training.yml          # Daily retraining workflow

```

## Setup & Installation

**Prerequisites**

* Python 3.11 (Required for Hopsworks compatibility)
* Hopsworks Account
* GitHub Account

**Installation**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/AQI-Predictor.git
cd AQI-Predictor

```

2. Install the required packages:

```bash
pip install -r requirements.txt

```

3. Set up your environment variables. Create a `.streamlit/secrets.toml` file in the root directory:

```toml
HOPSWORKS_API_KEY = "your_hopsworks_api_key_here"

```

4. Run the Streamlit Dashboard locally:

```bash
streamlit run app.py

```

## Explainable AI (SHAP)

To avoid a "black box" scenario, this system integrates SHAP (SHapley Additive exPlanations).

During the validation phase of the daily training pipeline, the script uses the appropriate explainer (TreeExplainer or KernelExplainer) depending on which model won the tournament. It calculates the impact of each feature and saves a `shap_summary.png` file. This plot is embedded directly into the Streamlit dashboard, showing exactly which environmental factors pushed the prediction higher or lower.
