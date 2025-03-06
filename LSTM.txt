# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by a generic Python environment (specific Docker image reference removed)

import numpy as np  # Linear algebra
import pandas as pd  # Data processing, CSV file I/O
import os

# Input data files are available in a generic read-only input directory
# Listing files is commented out to avoid revealing specific file structure
# for dirname, _, filenames in os.walk('<INPUT_DIRECTORY>'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Import additional libraries
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed 
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(1)
tf.random.set_seed(1)

# Load data from a generic CSV file (specific path and filename removed)
df = pd.read_csv('<INPUT_FILE_PATH>') 
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1))
# Renaming columns with generic names
df.rename(columns={'<ORIGINAL_TARGET_COLUMN>': 'Target'}, inplace=True)

# Split into train and test sets
train_size = int(len(df) * 0.95)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

print(train.shape, test.shape)

# Scale the target variable
scaler = StandardScaler()
scaler = scaler.fit(train[['Target']])
train['Target'] = scaler.transform(train[['Target']]) 
test['Target'] = scaler.transform(test[['Target']])

# Function to create sequences
def create_seq(X, y, time_steps=1): 
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values 
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 30
# Reshape to [samples, time_steps, n_features] 
X_train, y_train = create_seq(train[['Target']], train.Target, TIME_STEPS)
X_test, y_test = create_seq(test[['Target']], test.Target, TIME_STEPS)

print(X_train.shape)
print(X_test.shape)

# Define the LSTM autoencoder model
model = keras.Sequential() 
model.add(keras.layers.LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(keras.layers.Dropout(rate=0.2)) 
model.add(keras.layers.RepeatVector(n=X_train.shape[1])) 
model.add(keras.layers.LSTM(units=64, return_sequences=True)) 
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))

model.compile(loss='mae', optimizer='adam')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, 
                    validation_split=0.1, shuffle=False)

# Plot training history
plt.plot(history.history['loss'], label='Training loss') 
plt.plot(history.history['val_loss'], label='Validation loss') 
plt.legend()

# Evaluate the model
model.evaluate(X_test, y_test)

# Calculate reconstruction loss
X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
sns.distplot(train_mae_loss, bins=50, kde=True)

X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

# Define anomaly threshold
mean_loss = np.mean(test_mae_loss)
std_loss = np.std(test_mae_loss)
THRESHOLD = mean_loss + 2 * std_loss  

# Create a DataFrame for test scores
test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index) 
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold 
test_score_df['Target'] = test[TIME_STEPS:].Target

# Plot loss vs threshold
plt.plot(test_score_df.index, test_score_df.loss, label='Loss') 
plt.plot(test_score_df.index, test_score_df.threshold, label='Threshold') 
plt.xticks(rotation=25)
plt.legend()

# Identify and transform anomalies
anomalies = test_score_df[test_score_df.anomaly == True]
anomalies_close = anomalies[["Target"]].values.flatten()
anomalies_close = anomalies_close.reshape(-1, 1)  # Reshape to a 2D array
anomalies_close = scaler.inverse_transform(anomalies_close)

# Plot target values with anomalies
plt.plot(test[TIME_STEPS:].index, 
         scaler.inverse_transform(test[TIME_STEPS:][["Target"]]), 
         label='Target')

anomalies_close = anomalies_close.flatten()
sns.scatterplot(x=anomalies.index, 
                y=anomalies_close, 
                color=sns.color_palette()[3], label='Anomaly')

plt.xticks(rotation=25) 
plt.legend()

# Add dates to test_score_df and merge with original data
start_date = df['Date'].min()  # Or choose any starting date
test_score_df['Date'] = pd.date_range(start=start_date, periods=len(test_score_df), freq='D')
merged_df = pd.merge(df, test_score_df[['Date', 'Target', 'anomaly']], on='Date', how='left')
merged_df.rename(columns={'Target_x': 'Target', 'Target_y': 'Predicted_Target'}, inplace=True)

# Plot target values over time with anomalies
plt.figure(figsize=(14, 7))
plt.plot(merged_df['Date'], merged_df['Target'], label='Target', color='b')
anomalies = merged_df[merged_df['anomaly'] == True]
plt.scatter(anomalies['Date'], anomalies['Target'], color='r', label='Anomaly', zorder=5)

plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Target')
plt.title('Target Values and Anomalies over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()