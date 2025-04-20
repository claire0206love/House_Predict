# 匯入所需套件
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 模型與工具
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 可選：使用 XGBoost（需要安裝）
# pip install xgboost
from xgboost import XGBRegressor

# 1. 載入資料
df = pd.read_csv("House_Predict")  # <<--- 這裡換成你的資料檔案名稱

# 2. 初步查看資料
print("前幾筆資料：")
print(df.head())
print("\n資料基本資訊：")
print(df.info())
print("\n缺失值概況：")
print(df.isnull().sum())

# 3. 處理缺失值（簡單處理為填補中位數，視情況可更細緻）
df = df.fillna(df.median(numeric_only=True))

# 4. 分類欄位轉換（如有文字類別欄位）
df = pd.get_dummies(df)

# 5. 切分特徵與目標（假設「Price」是要預測的房價欄位）
X = df.drop("Price", axis=1)
y = df["Price"]

# 6. 切分訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 建立模型（你可以選 LinearRegression 或 XGBRegressor）
model = XGBRegressor()
model.fit(X_train, y_train)

# 8. 預測與評估
y_pred = model.predict(X_test)
print("\n模型評估：")
print(f"MAE（平均絕對誤差）：{mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE（均方根誤差）：{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²（決定係數）：{r2_score(y_test, y_pred):.2f}")

# 9. 可視化預測結果
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("實際房價")
plt.ylabel("預測房價")
plt.title("實際 vs 預測房價")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 對角線
plt.grid(True)
plt.show()
