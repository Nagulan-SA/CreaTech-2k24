import numpy as np
import pandas as pd
#import pip
#pip.main(['install','tensorflow'])
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
import datetime

df = pd.read_csv("C:/Users/Nagulan/Desktop/CreaTech 2k24/CreaTech-2k24/Material-OHLC.csv")

cleaner = ['Open','High','Low','Prev.close','ltp','Close','vwap','52W H','52W L','Volume','Value','No of trades']

for i in range(0, len(df['Date'])):
    df['Date'][i] = datetime.datetime.strptime(df['Date'][i], "%d-%b-%y").strftime("%Y-%m-%d")

for j in range(0, len(df)):
    for k in cleaner:
        df[k][j] = float(df[k][j].replace(',','')) 
'''
plt.figure(figsize=(13,7))
sns.countplot(x="Material", data=df, palette='hls')
plt.show()'''

'''product_column = 'Material'
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

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Feature engineering: Creating lag features
#df['lag1'] = df['Close'].shift(1)
#df['lag2'] = df['Close'].shift(2)

test_result = adfuller(df['Close'])

def adfuller_test(sales):
    result = adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value, label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

adfuller_test(df['Close'])

df['Seasonal First Difference']=df['Close']-df['Close'].shift(12)

adfuller_test(df['Seasonal First Difference'].dropna())

#df['Seasonal First Difference'].plot()
#plt.show()

'''print(df['Close'])

autocorrelation_plot(df['Close'])
plt.show()'''

'''Historical closing price analysis for each material'''

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
    plt.show()

'''Feature engineering to analyze and select most important feature'''

#Lasso Regression for Feature Selection
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# Separate features and target variable
X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Prev.close', 'Value']]
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
X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Prev.close', 'Value']]
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

'''Seasonal-trend decomposition'''
'''
from statsmodels.tsa.seasonal import seasonal_decompose

materials = df['Material'].unique()

for m in materials:
    material_data = df[df['Material'] == m]
    
    # Decompose time series data to visualize trends, seasonality, and residuals
    decomposition = seasonal_decompose(material_data['Close'], model='additive', period=365)
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

'''Forecasting using Long-Short Term Memory(LSTM) model'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_sequences.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_sequences, train[sequence_length:], epochs=100, batch_size=32)

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
plt.plot(predictions, color='red', label='Forecast')
plt.title(f'{cast.upper()}\nLSTM Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Close Price (in INR)')
plt.legend()
plt.show()