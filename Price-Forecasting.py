import numpy as np
import pandas as pd
#import pip
#pip.main(['install','scikit-learn'])
import seaborn as sns
import matplotlib.pyplot as plt
#from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
import datetime

df = pd.read_csv("C:/Users/Nagulan/Desktop/CreaTech 2k24/CreaTech-2k24/Material-OHLC.csv")
print(df)
print(df.isnull().sum())

cleaner = ['Open','High','Low','Prev.close','ltp','Close','vwap','52W H','52W L','Volume','Value','No of trades']

for i in range(0, len(df['Date'])):
    df['Date'][i] = datetime.datetime.strptime(df['Date'][i], "%d-%b-%y").strftime("%Y-%m-%d")

for j in range(0, len(df)):
    for k in cleaner:
        df[k][j] = float(df[k][j].replace(',','')) 
             

print(df['Date'])

plt.figure(figsize=(13,7))
sns.countplot(x="Material", data=df, palette='hls')
plt.show()

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
df['lag1'] = df['Close'].shift(1)
df['lag2'] = df['Close'].shift(2)

'''# Train-test split
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# ARIMA model
model = ARIMA(train['Close'], order=(1, 1, 1))  # Adjust order based on grid search
fit_model = model.fit()'''

#test_result = adfuller(df['Close'])

'''print(df['Close'])

autocorrelation_plot(df['Close'])
plt.show()'''