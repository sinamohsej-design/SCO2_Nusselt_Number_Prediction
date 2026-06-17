# =========================
# 1. IMPORT LIBRARIES
# =========================

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

# =========================
# 2. REPRODUCIBILITY SEED
# =========================
np.random.seed(42)
tf.random.set_seed(42)

# =========================
# 3. LOAD DATA
# =========================

# Load experimental dataset (Excel file)
data = pd.read_excel("SCO2.xlsx")

# Input features (8 columns: pressure, temperature, Reynolds, etc.)
X = data.iloc[:, 0:8].values

# Output target: Local Nusselt number
y = data.iloc[:, 8].values.reshape(-1, 1)

# =========================
# 4. TRAIN-TEST SPLIT (IMPORTANT)
# =========================

# Split BEFORE scaling to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# 5. FEATURE SCALING
# =========================

# Normalize input features to [0,1] range
x_scaler = MinMaxScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

# Normalize output target (helps neural network stability)
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# =========================
# 6. BUILD MLP MODEL
# =========================

model = Sequential()

# Input + first hidden layer
model.add(Dense(
    64,
    activation='tanh',
    input_shape=(7,),
    kernel_regularizer=regularizers.l2(0.001) # L2 regularization
))

model.add(Dropout(0.2)) # Prevent overfitting

# Second hidden layer
model.add(Dense(
    32,
    activation='tanh',
    kernel_regularizer=regularizers.l2(0.001)
))

model.add(Dropout(0.2))

# Output layer (regression)
model.add(Dense(1, activation='linear'))

# Compile model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Show model architecture
model.summary()

# =========================
# 7. EARLY STOPPING (OVERFITTING CONTROL)
# =========================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)

# =========================
# 8. TRAIN MODEL
# =========================

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# 9. PREDICTION
# =========================

y_pred = model.predict(X_test)

# Convert back to original scale
y_pred_real = y_scaler.inverse_transform(y_pred)
y_test_real = y_scaler.inverse_transform(y_test)

# =========================
# 10. EVALUATION METRICS
# =========================

# R² Score (goodness of fit)
r2 = r2_score(y_test_real, y_pred_real)

# Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

# Mean Absolute Error
mae = mean_absolute_error(y_test_real, y_pred_real)

# Mean Absolute Percentage Error
mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100

# =========================
# 11. RESULTS OUTPUT
# =========================

print("\n================ RESULTS ================")
print(f"R2 Score = {r2:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"MAE = {mae:.4f}")
print(f"MAPE (%) = {mape:.2f}")
print("========================================")
