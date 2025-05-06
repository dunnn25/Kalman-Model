import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def apply_random_walk(time_series):

    kf_random_walk = KalmanFilter(
        initial_state_mean=time_series[0],
        initial_state_covariance=1.0,
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=0.1,
        observation_covariance=1.0
    )
    smoothed, _ = kf_random_walk.smooth(time_series)
    return smoothed.flatten()

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

    rw_smoothed = apply_random_walk(time_series)
    mse_rw = calculate_mse(time_series, rw_smoothed)

    plt.figure(figsize=(10, 6))
    plt.plot(ids, time_series, label='Dữ liệu gốc', alpha=0.5)
    plt.plot(ids, rw_smoothed, label=f'Random Walk (MSE: {mse_rw:.2f})', color='red')
    plt.xlabel('ID')
    plt.ylabel('Listening Time (phút)')
    plt.title(f'Mô hình Random Walk cho {podcast_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()