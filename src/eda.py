# src/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    return pd.read_csv(filepath, parse_dates=['Timestamp'])

def summary_statistics(df):
    return df.describe()

def data_quality_check(df):
    missing_values = df.isnull().sum()
    negative_values = df[['GHI', 'DNI', 'DHI']][(df['GHI'] < 0) | (df['DNI'] < 0) | (df['DHI'] < 0)]
    return missing_values, negative_values

def time_series_analysis(df):
    plt.figure(figsize=(15, 8))
    plt.plot(df['Timestamp'], df['GHI'], label='GHI')
    plt.plot(df['Timestamp'], df['DNI'], label='DNI')
    plt.plot(df['Timestamp'], df['DHI'], label='DHI')
    plt.plot(df['Timestamp'], df['Tamb'], label='Tamb')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Time Series Analysis')
    plt.show()

def correlation_analysis(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def wind_analysis(df):
    plt.figure(figsize=(15, 8))
    plt.plot(df['Timestamp'], df['WS'], label='Wind Speed')
    plt.plot(df['Timestamp'], df['WSgust'], label='Wind Gust')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Wind Speed and Gust over Time')
    plt.show()

def temperature_analysis(df):
    plt.figure(figsize=(15, 8))
    plt.plot(df['Timestamp'], df['TModA'], label='Module A Temperature')
    plt.plot(df['Timestamp'], df['TModB'], label='Module B Temperature')
    plt.plot(df['Timestamp'], df['Tamb'], label='Ambient Temperature')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Comparison')
    plt.show()

def histograms(df):
    df[['GHI', 'DNI', 'DHI']].hist(bins=20, figsize=(10, 8))
    plt.suptitle('Histograms of Solar Radiation Variables')
    plt.show()

    df[['Tamb', 'TModA', 'TModB']].hist(bins=20, figsize=(10, 8))
    plt.suptitle('Histograms of Temperature Variables')
    plt.show()

def box_plots(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[['GHI', 'DNI', 'DHI']])
    plt.title('Box Plot of Solar Radiation Variables')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[['Tamb', 'TModA', 'TModB']])
    plt.title('Box Plot of Temperature Variables')
    plt.show()

def scatter_plots(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Tamb', y='GHI', data=df)
    plt.title('Scatter Plot of GHI vs Tamb')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='WS', y='WSgust', data=df)
    plt.title('Scatter Plot of WS vs WSgust')
    plt.show()

def data_cleaning(df):
    #
    df_cleaned = df.dropna()
    df_cleaned = df_cleaned[(df_cleaned['GHI'] >= 0) & (df_cleaned['DNI'] >= 0) & (df_cleaned['DHI'] >= 0)]
    df_cleaned = df_cleaned.drop(columns=['Comments'])
    return df_cleaned

def detect_outliers(df,column):
    threshold=3
    # Detect outliers in a specified column using Z-score method.
    mean = df[column].mean()
    std_dev = df[column].std()
    z_scores = (df[column] - mean) / std_dev
    outliers = df[np.abs(z_scores) > threshold]
    return outliers