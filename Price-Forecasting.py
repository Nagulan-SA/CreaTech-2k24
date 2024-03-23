import numpy as np
import pandas as pd
import tensorflow as tf
#import pip
#pip.main(['install','pmdarima'])
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot

df = pd.read_csv("C:/Users/Nagulan/Desktop/CreaTech 2k24/CreaTech-2k24/Material-OHLC.csv")

'''Data Cleaning'''

import datetime

def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)

cleaner = ['Open','High','Low','Close','Adj Close','Volume']

df.dropna(inplace = True)

#for j in range(0, len(df)):
#    for k in cleaner:
#        df[k][j] = float(df[k][j].replace(',','')) 

df['Date'] = df['Date'].apply(str_to_datetime)
#df['Date'] = pd.to_datetime(df['Date'])
df.set_index(df['Date'], inplace = True)

'''Plotting Volume Comparison between materials'''
'''
product_column = 'Material'
sales_column = 'Volume'

plt.figure(figsize=(10,6))

for product in df[product_column].unique():
    product_data = df[df[product_column] == product]
    plt.plot(product_data[sales_column], label = product)

plt.xlabel('Materials')
plt.ylabel('Volume')
plt.title('Volume for different Materials')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()'''

'''Augmented Dickey-Fuller(ADF) Test for recording the stationary state'''

def adfuller_test(sales):
    result = adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value, label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:       # more negative, the stronger the rejection of null hypothesis
        print("Strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

adfuller_test(df['Close'])

#autocorrelation_plot(df['Close'])     # Correlation plot
#plt.show()

'''Historical closing price analysis for each material'''
'''
materials = df['Material'].unique()

for material in materials:
    material_data = df[df['Material'] == material]
    
    # Plot closing prices for the current material
    plt.figure(figsize=(13, 6))
    plt.plot(material_data.index, material_data['Close'], label=f'Closing Price - {material}',color='red')
    plt.title(f'Historical Prices for {material}')
    plt.xlabel('Year')
    plt.ylabel('Price (in INR)')
    plt.legend()
    plt.show()'''

'''Feature Engineering to analyze and select most important feature'''

#Lasso Regression for Feature Selection
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# Separate features and target variable
X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
y = df['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Lasso regression model
alpha = 0.01  # Adjust alpha based on your preference
lasso = Lasso(alpha=alpha)
lasso.fit(X_train, y_train)

# Get coefficients and print feature importance
coefficients = lasso.coef_
feature_importance = {feature: coef for feature, coef in zip(X_train.columns, coefficients)}

print("\nFeature Importance (Lasso Regression):\n")
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance}")

from sklearn.ensemble import RandomForestRegressor

# Separate features and target variable
X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
y = df['Close']

# Initialize Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Fit the model
rf_model.fit(X, y)

# Get feature importances
feature_importance = rf_model.feature_importances_

print("\nFeature Importance (Random Forest):\n")
for feature, importance in zip(X.columns, feature_importance):
    print(f"{feature}: {importance}")

'''Seasonal-trend Decomposition'''
'''
from statsmodels.tsa.seasonal import seasonal_decompose

materials = df['Material'].unique()

for m in materials:
    material_data = df[df['Material'] == m]
    
    # Decompose time series data to visualize trends, seasonality, and residuals
    decomposition = seasonal_decompose(material_data['Close'], model='additive', period = 365)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot the decomposed components
    plt.figure(figsize=(12, 8))

    plt.subplot(411)
    plt.plot(material_data['Close'], label='Original')
    plt.legend(loc='upper left')
    plt.title(f'{m}\nOriginal Time Series')

    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='upper left')
    plt.title('Trend Component')

    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='upper left')
    plt.title('Seasonality Component')

    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='upper left')
    plt.title('Residual Component')

    plt.tight_layout()
    plt.show()'''

'''Forecasting using Auto Regression Integrated Moving Average(ARIMA) model'''
'''
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

materials = df['Material'].unique()

for material in materials:
    material_data = df[df['Material'] == material]
    
    df_resampled = material_data.resample('B').mean(numeric_only= True)

    # Feature engineering: Create lag features
    df_resampled['lag1'] = df_resampled['Close'].shift(1)
    df_resampled['lag2'] = df_resampled['Close'].shift(2)

    # Train-test split
    train_size = int(len(df_resampled) * 0.8)
    train, test = df_resampled[:train_size], df_resampled[train_size:]

    train.dropna(inplace = True)
    test.dropna(inplace = True)

    # ARIMA model
    model = ARIMA(train['Close'], order=(1, 1, 1))
    fit_model = model.fit()

    # Forecast
    forecast = fit_model.forecast(steps=len(test))

    # Evaluate performance
    mse = mean_squared_error(test['Close'], forecast)
    print(f'Mean Squared Error: {mse}')
    
    # Plot actual vs. forecasted
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Train')
    plt.plot(test.index, test['Close'], label='Test')
    plt.plot(test.index, forecast, label='Forecast', linestyle='dashed')
    plt.title(f'Price Forecasting using ARIMA\nfor {material}\nMedian at {np.nanmean(forecast)}')
    plt.axhline(y=np.nanmean(forecast), color='violet', linestyle='--', linewidth=3, label='Avg')
    plt.xlabel('Date')
    plt.ylabel('Close Price (in INR)')
    plt.legend()
    plt.show()

    future_date = input('Enter the date you want to predict : ')
    end_date = input("Enter the end date : ")
    future = pd.date_range(start = pd.to_datetime(future_date), end = pd.to_datetime(end_date))
    pred = fit_model.predict(len(df_resampled), len(df_resampled) + len(future) - 1, dynamic = True, typ = 'levels').rename("Predictions")
    pred.index = future
    print(pred)
    pred.plot(figsize=(12,6))
    plt.show()'''
'''
from pmdarima import auto_arima
import warnings

warnings.filterwarnings("ignore")

stepwise_fit = auto_arima(df['Close'], trace = True, suppress_warnings = True)
stepwise_fit.summary()

from statsmodels.tsa.arima.model import ARIMA

df = df[df['Material'] == "SAIL STEEL"]

train_len = int(len(df) * 0.8)

train, test = df[:train_len], df[train_len:]
#train.dropna(inplace = True )
#test = df.iloc[-300:]
#test.dropna(inplace = True)

#print(df.shape)
#print(train.shape, test.shape)

model = ARIMA((train['Close']), order = (0,1,0))
model = model.fit()
model.summary()

start = len(train)
end = len(train) + len(test) - 1
#index_date = pd.date_range("2024-03-22","2024-03-29")
pred = model.predict(start = start, end = end, typ = 'levels')
print(pred)

pred.plot(label = 'Forecast', legend = True)
test['Close'].plot(legend = True)
plt.show()'''

'''Forecasting using Long-Short Term Memory(LSTM) model'''

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

materials = df['Material'].unique()
print(materials)             # for user reference while browsing

while True:
    try:
        cast = input("Price Forecast for :")
        if cast.upper() in materials:
            break
    except ValueError:
        print("Select from these available data :",materials)
        continue

material_data = df[df['Material'] == cast.upper()]

target_variable = 'Close'    # Variable with higher feature importance is 'Close'

# Extract the target variable
data = material_data[[target_variable]].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[:train_size], scaled_data[train_size:]

# Function to create sequences for LSTM
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        sequences.append(seq)
    return np.array(sequences)

# Define sequence length
sequence_length = 20              # We may need to tune this parameter

# Create sequences for training
train_sequences = create_sequences(train, sequence_length)

# Build LSTM model

model = Sequential()
model.add(LSTM(units=100, activation = 'relu', return_sequences=True, input_shape=(train_sequences.shape[1], 1)))
model.add(LSTM(units=100))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_sequences, train[sequence_length:], epochs=100, batch_size=16)
print(model.summary())

loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.show()

# Create sequences for testing
test_sequences = create_sequences(test, sequence_length)

# Make predictions
predictions = model.predict(test_sequences)

# Inverse transform to get original scale
predictions = scaler.inverse_transform(predictions)
test_original = scaler.inverse_transform(test[sequence_length:])

# Calculate evaluation metrics (e.g., MAE, MSE)
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test_original, predictions)
mse = mean_squared_error(test_original, predictions)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

# Plotting the forecast
#plt.figure(figsize=(12, 8))
#plt.plot(test_original, label='Observed')
#plt.plot(predictions, color='red', linestyle = '-', label='Forecast')
#plt.axhline(y=np.nanmean(predictions), color='violet', linestyle='--', linewidth=3, label='Avg')
#plt.title(f'{cast.upper()}\nLSTM Forecast')
#plt.xlabel('Time Steps')
#plt.ylabel('Close Price (in INR)')
#plt.legend()
#plt.show()

date_index = df.index.tolist()

from copy import deepcopy

recursive_predictions = []
recursive_dates=np.concatenate([date_index[:train_size], date_index[train_size:]])
last_window = deepcopy(train_sequences[-sequence_length:])

for target_date in recursive_dates:
    next_prediction=model.predict(last_window)
    recursive_predictions.append(next_prediction)
    next_prediction = np.array(next_prediction).reshape(1,sequence_length,1)
    last_window = last_window[1:]
    last_window = np.concatenate([last_window, next_prediction], axis=0)

recursive_predictions = np.array(recursive_predictions)

# Plotting the forecast
plt.figure(figsize=(12, 8))
plt.plot(date_index[-len(test_original):], test_original, label = 'Observed')
plt.plot(date_index[-len(predictions):], predictions, color='red', label = 'Predicted')
plt.plot(recursive_dates, recursive_predictions.reshape(recursive_dates.shape[0],1), label = 'Forecast')
plt.axhline(y=np.nanmean(predictions), color='violet', linestyle='--', linewidth=3, label='Avg')
plt.title(f'{cast.upper()}\nLSTM Forecast\nMedian at {np.nanmean(predictions)} INR')
plt.xlabel('Date')
plt.ylabel('Close Price (in INR)')
plt.legend()
plt.show()

'''
df = pd.read_csv("C:/Users/Nagulan/Desktop/CreaTech 2k24/CreaTech-2k24/Material-OHLC.csv")
df['Date'] = df['Date'].apply(str_to_datetime)

materials = df['Material'].unique()
print(materials)             # for user reference while browsing

while True:
    try:
        cast = input("Price Forecast for :")
        if cast.upper() in materials:
            break
    except ValueError:
        print("Select from these available data :",materials)
        continue

df = material_data[['Date', target_variable]]

df.index = df.pop('Date')

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
  first_date = str_to_datetime(first_date_str)
  last_date  = str_to_datetime(last_date_str)

  target_date = first_date
  
  dates = []
  X, Y = [], []

  last_time = False
  while True:
    df_subset = dataframe.loc[:target_date].tail(n+1)
    
    if len(df_subset) != n+1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      return

    values = df_subset['Close'].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
    
    if last_time:
      break
    
    target_date = next_date

    if target_date == last_date:
      last_time = True
    
  ret_df = pd.DataFrame({})
  ret_df['Target Date'] = dates
  
  X = np.array(X)
  for i in range(0, n):
    X[:, i]
    ret_df[f'Target-{n-i}'] = X[:, i]
  
  ret_df['Target'] = Y

  return ret_df

windowed_df = df_to_windowed_df(df, '2015-01-06', '2024-03-20', n=3)

def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float64), Y.astype(np.float64)

dates, X, y = windowed_df_to_date_X_y(windowed_df)

train_size_80 = int(len(dates) * .8)
train_size_90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:train_size_80], X[:train_size_80], y[:train_size_80]
dates_val, X_val, y_val = dates[train_size_80:train_size_90], X[train_size_80:train_size_90], y[train_size_80:train_size_90]
dates_test, X_test, y_test = dates[train_size_90:], X[train_size_90:], y[train_size_90:]

from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers

model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

print(model.summary())

train_predictions = model.predict(X_train).flatten()
val_predictions = model.predict(X_val).flatten()
test_predictions = model.predict(X_test).flatten()

from copy import deepcopy

recursive_predictions = []
recursive_dates=np.concatenate([dates_val, dates_test])
last_window = deepcopy(X_train[-1])

for target_date in recursive_dates:
    next_prediction=model.predict([last_window])
    recursive_predictions.append(next_prediction)
    last_window=np.concatenate([last_window[-2:],next_prediction])

print(recursive_predictions)

plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
#plt.plot(recursive_dates, recursive_predictions)
plt.legend(['Training Predictions', 
            'Training Observations',
            'Validation Predictions', 
            'Validation Observations',
            'Testing Predictions', 
            'Testing Observations',
            'Recursive Predictions'])
plt.show()'''