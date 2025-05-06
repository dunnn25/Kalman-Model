import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def apply_local_linear_trend(time_series):

    kf_linear_trend = KalmanFilter(
        initial_state_mean=[time_series[0], 0],  # [level, trend]
        initial_state_covariance=np.eye(2) * 1.0,
        transition_matrices=[[1, 1], [0, 1]],  # [level_t = level_{t-1} + trend_{t-1}, trend_t = trend_{t-1}]
        observation_matrices=[[1, 0]],  # Chỉ quan sát level
        transition_covariance=np.diag([0.1, 0.01]),  # Nhiễu cho level và trend
        observation_covariance=1.0
    )
    smoothed, _ = kf_linear_trend.smooth(time_series)
    return smoothed[:, 0]

def calculate_mse(original, predicted):

    return mean_squared_error(original, predicted)

if __name__ == '__main__':
    # Ví dụ sử dụng (có thể bỏ qua hoặc sửa đổi khi import)
    df = pd.read_csv('du_lieu_chi_co_Thursday.csv')
    podcast_name = 'Music Matters'
    data_filtered = df[df['Podcast_Name'] == podcast_name].copy()
    data_filtered = data_filtered.sort_values('id')
    data_filtered['Listening_Time_minutes'].fillna(data_filtered['Listening_Time_minutes'].mean(), inplace=True)
    time_series = data_filtered['Listening_Time_minutes'].values
    ids = data_filtered['id'].values

    lt_smoothed = apply_local_linear_trend(time_series)
    mse_lt = calculate_mse(time_series, lt_smoothed)

    plt.figure(figsize=(10, 6))
    plt.plot(ids, time_series, label='Dữ liệu gốc', alpha=0.5)
    plt.plot(ids, lt_smoothed, label=f'Local Linear Trend (MSE: {mse_lt:.2f})', color='blue')
    plt.xlabel('ID')
    plt.ylabel('Listening Time (phút)')
    plt.title(f'Mô hình Local Linear Trend cho {podcast_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()