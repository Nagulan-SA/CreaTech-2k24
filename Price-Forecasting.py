import numpy as np
import pandas as pd
#import pip
#pip.main(['install','tensorflow'])
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
import datetime

df = pd.read_csv("C:/Users/Nagulan/Desktop/CreaTech 2k24/CreaTech-2k24/Material-OHLC.csv")

'''Data Cleaning'''

cleaner = ['Open','High','Low','Prev.close','ltp','Close','vwap','52W H','52W L','Volume','Value','No of trades']

for i in range(0, len(df['Date'])):
    df['Date'][i] = datetime.datetime.strptime(df['Date'][i], "%d-%b-%y").strftime("%Y-%m-%d")

for j in range(0, len(df)):
    for k in cleaner:
        df[k][j] = float(df[k][j].replace(',','')) 

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
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

'''Augmented Dickey-Fuller(ADF) Test for recording the statistical data'''

test_result = adfuller(df['Close'])

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

#plt.show()

df['Seasonal First Difference']=df['Close']-df['Close'].shift(12)

adfuller_test(df['Seasonal First Difference'].dropna())

#df['Seasonal First Difference'].plot()
#plt.show()

'''
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

'''Feature Engineering to analyze and select most important feature'''

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

'''Seasonal-trend Decomposition'''

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
    #plt.show()

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
#plt.show()

'''Demand Projection'''

from tensorflow.keras.layers import Dropout

materials = df['Material'].unique()
print(materials)             # for user reference while browsing

while True:
    try:
        predict = input("Demand projection for :")
        if predict.upper() in materials:
            break
    except ValueError:
        print("Select from these available data :",materials)
        continue

material_data = df[df['Material'] == predict.upper()]

# Define features and target variable
features = ['Open', 'High', 'Low', 'Close', 'Volume']
target_variable = 'Close'

# Combine features and target variable
selected_columns = features + [target_variable]

# Extract relevant data
data = material_data[selected_columns]

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Define seq_length (used during model training)
seq_length = 7

# Creating sequences for LSTM training
X, y = [], []
for i in range(len(scaled_data) - seq_length):
    X.append(scaled_data[i:i+seq_length, :-1])  # Exclude the target variable
    y.append(scaled_data[i+seq_length, -1])

X, y = np.array(X), np.array(y)

# Build LSTM model
model = Sequential()
model.add(Dropout(0.2))
# First LSTM layer with return_sequences=True
model.add(LSTM(units=100, activation='sigmoid', input_shape=(seq_length, X.shape[2]), return_sequences=True))
# Second LSTM layer with return_sequences=True
model.add(LSTM(units=100, activation='sigmoid', return_sequences=True))
# Third LSTM layer without return_sequences
model.add(LSTM(units=100, activation='sigmoid'))
# Output layer
model.add(Dense(units=1,activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
model.fit(X, y, epochs=100, batch_size=32)

while True:
    try:
        period = int(input("Enter the time period of demand prediction (in days): "))
        if period > 0:
            break
    except ValueError:
        print("Give a number (in integers)")
        continue

print(''' 
    Enter the specific timeline you want to look up on (frequency)
      
      Day - D          Year - Y            Month End - ME      Month Start - MS 
      Month - M        Business day - B    Weekly - W          Quarter end - Q\n
      ''')

while True:
    try:
        freq = input()
        if freq.upper() in ['D','Y','ME','MS','M','B','W','Q']:
            break
    except ValueError:
        print("Choose from the given frequencies")
        continue

future_dates = pd.date_range(start = df.index[-1], periods = period+1, freq = freq.upper())[1:]

# Creating a DataFrame with these future dates as index
future_data = pd.DataFrame(index=future_dates, columns=features)

future_data.fillna(0, inplace=True)
scaled_future_data = scaler.transform(future_data[features + [target_variable]])
# Reshape the features for LSTM input
scaled_future_data = scaled_future_data.reshape((1, len(features) + 1, period))
# Use the trained LSTM model to project future demand
future_predictions = model.predict(scaled_future_data)

# Manually scaling the predictions back to the original range
min_target = scaler.data_min_[-1]
max_target = scaler.data_max_[1]
future_predictions_original = future_predictions * (max_target - min_target) + min_target
# Create a DataFrame with appropriate date index
future_demand_df = pd.DataFrame(index=future_dates, columns=['Projected_Demand'])
future_demand_df['Projected_Demand'] = future_predictions_original.squeeze()

print('''
      Choose an option 
      
      1) Tabular presentation
      2) Graphical plot\n''')

while True:
    opted = input()

    if opted == '1':
        print(future_demand_df)
        break

    elif opted == '2':
        # Plotting the projected demand
        plt.figure(figsize=(12, 8))
        plt.plot(future_demand_df.index, future_demand_df['Projected_Demand'], label='Projected Demand', marker='o')
        plt.title(f'Projected Demand using LSTM\nfor {predict.upper()}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()

        observed_data = df[df["Material"] == predict.upper()]['Close']
        plt.plot(observed_data, label = 'Observed', marker = 'o', color = 'violet')
        plt.title(f'Observed data\nfor {predict.upper()}')
        plt.xlabel('Year')
        plt.ylabel('Closed price')
        plt.legend()
        plt.show()
        break

    else:
        continue