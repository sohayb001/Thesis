# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script:
# dataset = pandas.DataFrame(Country, Shipment ID, Year, Quarter, Month, Day)
# dataset = dataset.drop_duplicates()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

# Combine Year, Month, and Day into a Date column
dataset['Date'] = pd.to_datetime(dataset[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1))

# Choose a specific category (e.g., country, region) - replaced specific value with placeholder
category_data = dataset[dataset['<CATEGORY_COLUMN>'] == '<CATEGORY_VALUE>']

# Sort by Date
category_data = category_data.sort_values(by='Date')

# Feature engineering: Extract features like day of the week, month, etc.
category_data['Day_of_week'] = category_data['Date'].dt.dayofweek
category_data['Month'] = category_data['Date'].dt.month
category_data['Day_of_year'] = category_data['Date'].dt.dayofyear

# Use engineered features as input and a target column
X = category_data[['Day_of_week', 'Month', 'Day_of_year']]
y = category_data['<TARGET_COLUMN>']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

# Create the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Forecast future values (for 365 days)
forecast_features = pd.DataFrame({
    'Day_of_week': np.arange(0, 365) % 7,  # Mocking future days of the week
    'Month': 1,  # Placeholder for a fixed month (e.g., January)
    'Day_of_year': np.arange(1, 366)
})

forecast = model.predict(forecast_features)

# Plot the forecast
plt.figure(figsize=(20, 6))

# Plot historical data
plt.plot(category_data['Date'], category_data['<TARGET_COLUMN>'], label='Historical Data')

# Generate future dates for forecasting (365 days after the last date in the data)
future_dates = pd.date_range(category_data['Date'].iloc[-1], periods=366, freq='D')[1:]

# Plot forecast
plt.plot(future_dates, forecast, label='Forecast', color='red')

# Set title and labels
plt.title('Forecast Using XGBoost')
plt.xlabel('Date')
plt.ylabel('<TARGET_LABEL>')
plt.legend()

# Show plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()