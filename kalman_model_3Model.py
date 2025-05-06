import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv('du_lieu_chi_co_Thursday.csv')

# Lọc dữ liệu cho podcast "Music Matters"
podcast_name = 'Music Matters'
data_filtered = df[df['Podcast_Name'] == podcast_name].copy()

# Sắp xếp dữ liệu theo id (giả định id tăng dần theo thời gian)
data_filtered = data_filtered.sort_values('id')

# Xử lý giá trị thiếu trong cột Listening_Time_minutes
data_filtered['Listening_Time_minutes'].fillna(data_filtered['Listening_Time_minutes'].mean(), inplace=True)

# Chuẩn bị dữ liệu
time_series = data_filtered['Listening_Time_minutes'].values
ids = data_filtered['id'].values

# --- Mô hình 1: Random Walk ---
kf_random_walk = KalmanFilter(
    initial_state_mean=time_series[0],
    initial_state_covariance=1.0,
    transition_matrices=[1],
    observation_matrices=[1],
    transition_covariance=0.1,
    observation_covariance=1.0
)
rw_smoothed, _ = kf_random_walk.smooth(time_series)

# --- Mô hình 2: Local Linear Trend ---
kf_linear_trend = KalmanFilter(
    initial_state_mean=[time_series[0], 0],  # [level, trend]
    initial_state_covariance=np.eye(2) * 1.0,
    transition_matrices=[[1, 1], [0, 1]],  # [level_t = level_{t-1} + trend_{t-1}, trend_t = trend_{t-1}]
    observation_matrices=[[1, 0]],  # Chỉ quan sát level
    transition_covariance=np.diag([0.1, 0.01]),  # Nhiễu cho level và trend
    observation_covariance=1.0
)
lt_smoothed, _ = kf_linear_trend.smooth(time_series)
lt_smoothed = lt_smoothed[:, 0]  # Chỉ lấy level

# --- Mô hình 3: Smooth Trend ---
alpha = 0.9  # Hằng số làm mượt xu hướng
kf_smooth_trend = KalmanFilter(
    initial_state_mean=[time_series[0], 0],  # [level, trend]
    initial_state_covariance=np.eye(2) * 1.0,
    transition_matrices=[[1, 1], [0, alpha]],  # [level_t = level_{t-1} + trend_{t-1}, trend_t = alpha * trend_{t-1}]
    observation_matrices=[[1, 0]],  # Chỉ quan sát level
    transition_covariance=np.diag([0.1, 0.01]),  # Nhiễu cho level và trend
    observation_covariance=1.0
)
st_smoothed, _ = kf_smooth_trend.smooth(time_series)
st_smoothed = st_smoothed[:, 0]  # Chỉ lấy level

# Tính Mean Squared Error (MSE) để so sánh
mse_rw = mean_squared_error(time_series, rw_smoothed)
mse_lt = mean_squared_error(time_series, lt_smoothed)
mse_st = mean_squared_error(time_series, st_smoothed)

# Vẽ biểu đồ so sánh
plt.figure(figsize=(14, 8))
plt.plot(ids, time_series, label='Dữ liệu gốc (Listening Time)', alpha=0.5, marker='o', linestyle='--')
plt.plot(ids, rw_smoothed, label=f'Random Walk (MSE: {mse_rw:.2f})', color='red', linewidth=2)
plt.plot(ids, lt_smoothed, label=f'Local Linear Trend (MSE: {mse_lt:.2f})', color='blue', linewidth=2)
plt.plot(ids, st_smoothed, label=f'Smooth Trend (MSE: {mse_st:.2f})', color='green', linewidth=2)
plt.xlabel('ID (Thời gian)')
plt.ylabel('Listening Time (phút)')
plt.title(f'So sánh ba mô hình Kalman cho {podcast_name}')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Lưu biểu đồ
plt.savefig('kalman_three_models_comparison.png')
plt.show()

# In thông tin
print(f"Số lượng điểm dữ liệu: {len(time_series)}")
print(f"MSE Random Walk: {mse_rw:.2f}")
print(f"MSE Local Linear Trend: {mse_lt:.2f}")
print(f"MSE Smooth Trend: {mse_st:.2f}")
print(f"Giá trị Listening Time gốc (5 điểm đầu): {time_series[:5]}")
print(f"Giá trị Random Walk Smoothed (5 điểm đầu): {rw_smoothed[:5].flatten()}")
print(f"Giá trị Local Linear Trend Smoothed (5 điểm đầu): {lt_smoothed[:5]}")
print(f"Giá trị Smooth Trend Smoothed (5 điểm đầu): {st_smoothed[:5]}")