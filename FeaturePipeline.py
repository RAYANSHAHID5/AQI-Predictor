"""
fetches weather and pollutant data from AQICN,generates the features needed by
the model including timestamps and AQI movement over time and stores them in
Hopsworks.
Scheduled to run every hour as part of the CI/CD pipeline.
"""

import requests
import json
import pandas as pd
from datetime import datetime
import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

CITY = "A540745"
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
LOCAL_FILE = "dummy_data.json"
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT_NAME = "AQI_Predictor_RS"


def fetch_and_save_data(city, token):
    """Fetches raw pollutant data from the external AQICN API."""
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    response = requests.get(url)

    if response.status_code == 429:
        print("Rate limit exceeded (HTTP 429).")
        return None
    elif response.status_code != 200:
        print(f"Error: Failed to fetch data. Status code: {response.status_code}")
        return None

    data = response.json()
    if data['status'] == 'ok':
        with open(LOCAL_FILE, 'w') as f:
            json.dump(data, f)
        print(f"Live data fetched and saved to {LOCAL_FILE}")
        return data['data']
    else:
        print(f"API Error: {data['data']}")
        return None


def load_local_data():
    """Fallback mechanism for local testing to preserve API rate limits."""
    if not os.path.exists(LOCAL_FILE):
        print("No local data found. Fetch live data first.")
        return None

    with open(LOCAL_FILE, 'r') as f:
        data = json.load(f)
        return data['data']


def generate_features(raw_data):
    """
    Feature Engineering: Computes features from raw data, including
    time-based features (hour, day, month) for temporal pattern recognition.
    """
    if not raw_data:
        return None

    current_time = datetime.now()
    iaqi = raw_data.get('iaqi', {})

    feature_dict = {
        'timestamp': current_time,
        'target_aqi': raw_data.get('aqi'),
        'pm25': iaqi.get('pm25', {}).get('v', 0),
        'pm10': iaqi.get('pm10', {}).get('v', 0),
        'temp': iaqi.get('t', {}).get('v', 0),
        'humidity': iaqi.get('h', {}).get('v', 0),
        'wind_speed': iaqi.get('w', {}).get('v', 0),
        'hour': current_time.hour,
        'day': current_time.day,
        'month': current_time.month,
        'day_of_week': current_time.weekday()
    }

    return pd.DataFrame([feature_dict])


def upload_to_hopsworks(df):
    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(
        project=PROJECT_NAME,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()

    aqi_fg = fs.get_or_create_feature_group(
        name="lahore_aqi_features",
        version=2,
        primary_key=["timestamp"],
        description="AQI and weather features for DHA Lahore"
    )

    aqi_fg.insert(df)
    print("Successfully uploaded features to Hopsworks.")


if __name__ == "__main__":
    FETCH_LIVE = os.getenv("FETCH_LIVE", "true").lower() == "true"

    if FETCH_LIVE:
        raw_data = fetch_and_save_data(CITY, AQICN_TOKEN)
    else:
        raw_data = load_local_data()

    features_df = generate_features(raw_data)

    if features_df is not None:
        try:
            # Derived Feature: Calculate AQI change rate dynamically using previous state
            print("Fetching previous AQI state from Feature Store...")
            project = hopsworks.login(project=PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
            fs = project.get_feature_store()
            aqi_fg = fs.get_feature_group(name="lahore_aqi_features", version=2)

            last_df = aqi_fg.read()
            last_aqi = last_df.sort_values('timestamp').iloc[-1]['target_aqi']

            current_aqi = features_df['target_aqi'].iloc[0]
            features_df['aqi_change_rate'] = current_aqi - last_aqi
        except Exception as e:
            print("Could not fetch previous AQI. Defaulting change rate to 0.")
            features_df['aqi_change_rate'] = 0

        features_df['aqi_change_rate'] = features_df['aqi_change_rate'].astype('Int64')
        features_df['temp'] = features_df['temp'].astype('float64')

        upload_to_hopsworks(features_df)