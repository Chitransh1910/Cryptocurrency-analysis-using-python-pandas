import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
sns.set()

data = pd.read_csv("coin_Bitcoin.csv")
print("Shape of Dataset is: ",data.shape,"\n")
print(data.head())

data = pd.read_csv("coin_Ethereum.csv")
print("Shape of Dataset is: ",data.shape,"\n")
print(data.head())

data = pd.read_csv("Dogecoin.csv")
print("Shape of Dataset is: ",data.shape,"\n")
print(data.head())
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file into a DataFrame
df = pd.read_csv('coin_Bitcoin.csv')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' column as the index
df.set_index('Date', inplace=True)

# Plot the closing price of Bitcoin
plt.figure(figsize=(9, 6))
plt.plot(df['Close'])
plt.title('Bitcoin Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()

#ethereum

df = pd.read_csv('coin_Ethereum.csv')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' column as the index
df.set_index('Date', inplace=True)

# Plot the closing price of Bitcoin
plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.title('ethereum Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()
#litecoin

df = pd.read_csv('coin_Litecoin.csv')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' column as the index
df.set_index('Date', inplace=True)

# Plot the closing price of Bitcoin
plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.title('Dogecoin Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV file
df = pd.read_csv('coin_Bitcoin.csv')

# Drop unnecessary columns
df = df.drop(columns=['Date', 'Open', 'Close', 'High', 'Low', 'Volume'])

# Convert 'Price' column to numeric
df['Marketcap'] = pd.to_numeric(df['Marketcap'])

# Calculate daily returns
df['Daily_Return'] = df['Marketcap'].pct_change() * 100



# Plot histogram of daily returns
plt.hist(df['Daily_Return'], bins=15, edgecolor='green')
plt.xlabel('Daily Returns (%)')
plt.ylabel('Frequency')
plt.title('Histogram of Daily Returns of Bitcoin')
plt.show()

# Read data from CSV file
df = pd.read_csv('coin_Ethereum.csv')
# Plot histogram of daily returns
plt.hist(df['Daily_Return'], bins=15, edgecolor='green')
plt.xlabel('Daily Returns (%)')
plt.ylabel('Frequency')
plt.title('Histogram of Daily Returns of Ethereum')
plt.show()

# Read data from CSV file
df = pd.read_csv('coin_Dogecoin.csv')

# Plot histogram of daily returns
plt.hist(df['Daily_Return'], bins=15, edgecolor='green')
plt.xlabel('Daily Returns (%)')
plt.ylabel('Frequency')
plt.title('Histogram of Daily Returns of Dogecoin')
plt.show()

import pandas as pd
import mplfinance as mpf
import yfinance as yf
import matplotlib.pyplot as plt
# Get Bitcoin data from Yahoo Finance
df = yf.download(tickers='BTC-USD', period='1mo', interval='1d')

# Create candlestick chart
mpf.plot(df, type='candle',  figratio=(12,8), title='BTC-USD',savefig='Candlestick_3.png')
# Get Bitcoin data from Yahoo Finance
df = yf.download(tickers='BTC-USD', period='2y', interval='1d')

# Plot the volume as a bar chart
plt.figure(figsize=(12, 8))
plt.bar(df.index, df['Volume'], width=0.5, color='gray')
plt.title('Bitcoin Volume')
plt.xlabel('Date')
plt.ylabel('Volume (BTC)')
plt.tight_layout()
# Get Bitcoin data from Yahoo Finance
df = yf.download(tickers='BTC-USD', period='2y', interval='1d')

# Set up the figure and subplots using gridspec
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=[3, 1, 1, 1, 1])

# Plot the historical stock prices on the top subplot
ax1 = fig.add_subplot(gs[:3, :])
ax1.plot(df.index, df['Close'])
ax1.set_title('Historical stock prices of BTC-USD')

# Plot the trading volume on the bottom subplot
ax2 = fig.add_subplot(gs[3:, :])
ax2.bar(df.index, df['Volume'])
ax2.set_title('BTC-USD Volume')
plt.tight_layout()

mport pandas as pd
import mplfinance as mpf
import yfinance as yf
import matplotlib.pyplot as plt
# Get Bitcoin data from Yahoo Finance
df = yf.download(tickers='ETH-USD', period='1mo', interval='1d')

# Create candlestick chart
mpf.plot(df, type='candle',  figratio=(12,8), title='ETH-USD',savefig='Candlestick_3.png')
# Get Bitcoin data from Yahoo Finance
df = yf.download(tickers='ETH-USD', period='2y', interval='1d')

# Plot the volume as a bar chart
plt.figure(figsize=(12, 8))
plt.bar(df.index, df['Volume'], width=0.5, color='gray')
plt.title('Ethereum Volume')
plt.xlabel('Date')
plt.ylabel('Volume (ETH)')
plt.tight_layout()
# Get Bitcoin data from Yahoo Finance
df = yf.download(tickers='ETH-USD', period='2y', interval='1d')

# Set up the figure and subplots using gridspec
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=[3, 1, 1, 1, 1])

# Plot the historical stock prices on the top subplot
ax1 = fig.add_subplot(gs[:3, :])
ax1.plot(df.index, df['Close'])
ax1.set_title('Historical stock prices of ETH-USD')

# Plot the trading volume on the bottom subplot
ax2 = fig.add_subplot(gs[3:, :])
ax2.bar(df.index, df['Volume'])
ax2.set_title('ETH-USD Volume')
plt.tight_layout()


# Prediction tool for dogecoin.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
sns.set()

data = pd.read_csv("Dogecoin.csv")
print("Shape of Dataset is: ",data.shape,"\n")
print(data.head())


data.dropna()
plt.figure(figsize=(10, 4))
plt.title("DogeCoin Price INR")
plt.xlabel("Date")
plt.ylabel("Close")
plt.plot(data["Close"])
plt.show()

from autots import AutoTS


model = AutoTS(forecast_length=10, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
 
prediction = model.predict()
forecast = prediction.forecast
print("DogeCoin Price Prediction")
print(forecast)
