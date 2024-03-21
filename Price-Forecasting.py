import numpy as np
import pandas as pd
#import pip
#pip.main(['install','pmdarima'])
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot

df = pd.read_csv("C:/Users/Nagulan/Desktop/CreaTech 2k24/CreaTech-2k24/Material-OHLC.csv")

'''Data Cleaning'''

cleaner = ['Open','High','Low','Close','Adj Close','Volume']

df.dropna(inplace = True)

#for j in range(0, len(df)):
#    for k in cleaner:
#        df[k][j] = float(df[k][j].replace(',','')) 

df['Date'] = pd.to_datetime(df['Date'])
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
    plt.show()
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
'''
from keras.models import Sequential
from keras.layers import LSTM, Dense
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

print(model.summary())

# Train the model
model.fit(train_sequences, train[sequence_length:], epochs=100, batch_size=64)

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
plt.figure(figsize=(12, 8))
plt.plot(test_original, label='Observed')
plt.plot(predictions, color='red', linestyle = '-', label='Forecast')
plt.axhline(y=np.nanmean(predictions), color='violet', linestyle='--', linewidth=3, label='Avg')
plt.title(f'{cast.upper()}\nLSTM Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Close Price (in INR)')
plt.legend()
plt.show()'''
