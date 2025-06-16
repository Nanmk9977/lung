import pandas as pd
import numpy as np

def classify_patient_condition(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)

    # Rename for consistency
    df.rename(columns=lambda x: x.strip().lower(), inplace=True)
    if 'heart rate (bpm)' in df.columns:
        df.rename(columns={'heart rate (bpm)': 'heart_rate'}, inplace=True)
    if 'blood oxygen level (%)' in df.columns:
        df.rename(columns={'blood oxygen level (%)': 'spo2'}, inplace=True)

    # Clean the data
    df = df[['heart_rate', 'spo2']].dropna()
    df = df[(df['heart_rate'] >= 40) & (df['heart_rate'] <= 180)]
    df = df[(df['spo2'] >= 85) & (df['spo2'] <= 100)]

    if len(df) < 2:
        return "Insufficient Data"

    # Calculate trends
    hr_trend = np.sign(df['heart_rate'].iloc[-1] - df['heart_rate'].iloc[0])
    spo2_trend = np.sign(df['spo2'].iloc[-1] - df['spo2'].iloc[0])

    # Decision Logic
    if hr_trend <= 0 and spo2_trend >= 0:
        return "Vital Trends is Normal"
    elif hr_trend > 0 and spo2_trend < 0:
        return "Vital Trends Indicate Instability"
    else:
        return "Vital Trends is Normal"

