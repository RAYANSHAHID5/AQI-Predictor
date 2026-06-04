import requests
import pandas as pd
from datetime import datetime
import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT_NAME = "AQI_Predictor_RS"

LAT = 31.5256
LON = 74.4361
PAST_DAYS = 90 # Upgraded to 90 days (~2160 rows)

def fetch_historical_data():
    print(f"🌍 Fetching {PAST_DAYS} days of historical data from Open-Meteo...")
    aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}&hourly=pm10,pm2_5,us_aqi&timezone=auto&past_days={PAST_DAYS}"
    aq_data = requests.get(aq_url).json()

    wx_url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=auto&past_days={PAST_DAYS}"
    wx_data = requests.get(wx_url).json()

    aq_df = pd.DataFrame(aq_data['hourly'])
    wx_df = pd.DataFrame(wx_data['hourly'])
    df = pd.merge(aq_df, wx_df, on='time')

    df = df.rename(columns={
        'time': 'timestamp', 'us_aqi': 'target_aqi', 'pm2_5': 'pm25',
        'pm10': 'pm10', 'temperature_2m': 'temp',
        'relative_humidity_2m': 'humidity', 'wind_speed_10m': 'wind_speed'
    })

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.dropna()

    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.weekday

    # --- NEW FEATURE: AQI Change Rate ---
    df = df.sort_values('timestamp')
    df['aqi_change_rate'] = df['target_aqi'].diff().fillna(0)

    # Type casting
    int_columns = ['target_aqi', 'pm25', 'pm10', 'humidity', 'wind_speed', 'hour', 'day', 'month', 'day_of_week', 'aqi_change_rate']
    for col in int_columns:
        df[col] = df[col].round().astype('Int64')
    df['temp'] = df['temp'].astype('float64')

    print(f"✅ Prepared {len(df)} rows!")
    return df

def upload_to_hopsworks(df):
    """Uploads the historical dataset to your Hopsworks Feature Group."""
    print("🚀 Connecting to Hopsworks...")
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

    print(f"Uploading {len(df)} rows to the cloud... this might take a minute!")
    aqi_fg.insert(df)
    print("✅ Backfill Complete! Your database is now primed for Phase 2.")


if __name__ == "__main__":
    historical_df = fetch_historical_data()

    # Show a quick preview of the data types to verify they are all Int64/float64 now
    print("\nData Schema Verification:")
    print(historical_df.dtypes)
    print("-" * 30)

    upload_to_hopsworks(historical_df)
