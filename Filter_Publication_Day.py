import pandas as pd

# Đọc dữ liệu từ file CSV
df = pd.read_csv("train.csv")

# Lọc ra các dòng chỉ có 'Friday' trong cột Publication_Day
df_friday = df[df['Publication_Day'] == 'Thursday']

# (Tuỳ chọn) Lưu ra file mới
df_friday.to_csv("du_lieu_chi_co_Thursday.csv", index=False)

print("Đã lọc xong! Dữ liệu chỉ còn lại các dòng 'Thursday'.")