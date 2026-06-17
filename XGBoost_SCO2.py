# =========================
# 1. IMPORT LIBRARIES
# =========================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor

# =========================
# 2. LOAD DATA
# =========================

data = pd.read_excel("SCO2.xlsx")

# 8 input features
X = data.iloc[:, 0:8].values

# 1 target (Nusselt number)
y = data.iloc[:, 8].values

# =========================
# 3. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# 4. BUILD XGBOOST MODEL
# =========================

model = XGBRegressor(
    n_estimators=500,      # number of trees
    learning_rate=0.05,    # step size
    max_depth=6,           # tree depth (controls complexity)
    subsample=0.8,         # prevents overfitting
    colsample_bytree=0.8,  # feature sampling
    random_state=42
)

# =========================
# 5. TRAIN MODEL
# =========================

model.fit(X_train, y_train)

# =========================
# 6. PREDICTION
# =========================

y_pred = model.predict(X_test)

# =========================
# 7. EVALUATION METRICS
# =========================

r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

mae = mean_absolute_error(y_test, y_pred)

mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

# =========================
# 8. RESULTS OUTPUT
# =========================

print("\n================ RESULTS ================")
print(f"R2 Score = {r2:.4f}")
print(f"RMSE     = {rmse:.4f}")
print(f"MAE      = {mae:.4f}")
print(f"MAPE     = {mape:.2f}%")
print("========================================")
