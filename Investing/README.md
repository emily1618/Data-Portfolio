# Ichimoku Cloud and Optimal Portfolio
##### Author: Emi Ly

##### Date: Feb 18, 2022

##### [Tableau Dashboard]-Coming Soon
#

### INTRODUCTION

This project is not set out to recommend any stocks. I'm not a trader, not even close. However, it is my interest to learn more about how to read charts and using technical indicators. I learned the ichimoku code from Derek Banas, an Udemy instructor. He is an awesome coder and I will be continuing learning from his videos. I recommend anyone interested in the same topic NOT to just copy the code. It helps to type out all the code, google/stackoverflow anything you don't understand, and most importantly, google the concept behind ichimoku cloud, moving average, etc. Once you have a more than a basic understanding, try to use the code to build your own portfolio and use the ichimoku cloud to test trades. I built upon Derek Banas original ichimoku code. I added the EDA, modified the ichimoku code and run it differently for the optimal portfolio based on my preference. I only scratched the surface and there are so much more to learn! 

The code is seperated into 3 parts:
### 📊 [EDA on NYSE and NASDAQ](#eda-on-nyse-and-nasdaq)
### 🌦 [Ichimoku Cloud](#ichimoku-cloud)
### 💯 [Finding a Optimal Portfolio](#finding-a-optimal-portfolio)

Dataset:
- From `import yfinance as yf` and https://www.nasdaq.com/market-activity/stocks/screener
- I only picked the top 300 based on market cap. If you have time, please run on all of the NASDAQ tickers!
- I used a 5 year timeframe. To enhance your analysis, please also run a 6 months and 1 year comparison.




## EDA on NYSE and NASDAQ
We will import the necessary libraries:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as datetime
import time
import os
import seaborn as sns
```

Set a start and end date:
```
S_DATE = "2017-02-01"
E_DATE = "2023-02-01"
S_DATE_DT = pd.to_datetime(S_DATE)
E_DATE_DT = pd.to_datetime(E_DATE)
```

Downloading the data from yfinance and saving to csv (only do this way if you want to manually pick your own tickers):
```
tickers = ['AMZN', 'COST', 'ADBE', 'QQQ', 'NVDA', 'VUG', 'VGT', 'VOO', 'SPY', 'ARKK', 'ADBE','NVDA', 'AMD', 'MSFT', 'DIS', 'WMT', 'TGT', 'HD']

for ticker in tickers:
  stock = yf.Ticker(ticker)
  df = stock.history(period="5y")
  df.sort_index(axis = 0.,  ascending=False)
  df.to_csv(ticker + '.csv')
```

Read the data
```
nasdaq = pd.read_csv("/content/Nasdaq.csv")
nasdaq.head(3)
```
![1](https://user-images.githubusercontent.com/62857660/154596067-179b2ee5-3483-4a30-89a6-eeb89ade275b.jpg)

Check for interesting data using `nasdaq.describe()`. From there I noticed that the IPO Year is float64 due to NaN. It may be better to convert to a integer after you fix the NaN data. 

![9](https://user-images.githubusercontent.com/62857660/154600055-ac66b924-5848-476d-8a82-8851177b51ab.jpg)



2021 is the year with the most IPO. Significantly more!
```
ipo_year = nasdaq['IPO Year'].value_counts().sort_values(ascending=False)
ax = ipo_year.plot(kind='barh', figsize=(25, 10), color='#86bf91', zorder=2, width=0.85)
```
![2](https://user-images.githubusercontent.com/62857660/154597871-ed2c2eb4-d7af-48db-9937-7af614eaff0f.png)

Using `nasdaq.loc[nasdaq['IPO Year'] == 2021]` to see some of the companies that IPO in 2021.
![3](https://user-images.githubusercontent.com/62857660/154598065-daaa00b8-3d69-4e4f-8678-c694c6ea3b20.jpg)

Check to see how much data contains missing value. IPO Year contains 40% of the missing data, which is alot! 
```
percent_missing = nasdaq.isnull().sum() * 100 / len(nasdaq)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df.sort_values(by=['percent_missing'], ascending=False)

#duplicated values
duplicated_value_df = nasdaq.duplicated(keep=False).value_counts(normalize=True) * 100
```
![4](https://user-images.githubusercontent.com/62857660/154598907-d79f8be5-03a1-45bf-b3a4-2a6f44179abb.jpg)

Checking to see which company has the IPO year 1972 as that's the **earliest year** for IPO:

```nasdaq.loc[nasdaq['IPO Year'] == 1972]```

![5](https://user-images.githubusercontent.com/62857660/154599781-b2d6adc9-4920-41a5-a380-cf9b67c02152.jpg)


Checking to see which company has the **most market cap**: 

```nasdaq.loc[nasdaq['Market Cap'] == 3.000000e+12]```

![6](https://user-images.githubusercontent.com/62857660/154599785-2dd1e3d6-b5af-4a9a-b1f6-c34d719e502a.jpg)


Checking to see which company has the **max net change**: 

```nasdaq.loc[nasdaq['Net Change'] == 1570]```

![7](https://user-images.githubusercontent.com/62857660/154599794-dffbe2bd-cbcb-4730-a13e-6754500d1f6e.jpg)

Checking to see which company has the **least net change**:

```nasdaq.loc[nasdaq['Net Change'] == -86.16]```

![8](https://user-images.githubusercontent.com/62857660/154599796-fa937e84-74e3-4a0e-94d8-f610ba618767.jpg)

Checking to see the make up of sectors:
```
print(nasdaq['Sector'].value_counts(normalize=True)*100
```
![2](https://user-images.githubusercontent.com/62857660/154600608-d19f8b30-a738-4e90-a97e-feb7beef571a.png)

Each sector has different industries: `nasdaq['Industry'].groupby(nasdaq['Sector']).value_counts()`
![3](https://user-images.githubusercontent.com/62857660/154601922-228a77c5-e008-488a-8e6c-10d3f5298780.jpg)

Top 20 represented countries: 
```
country = nasdaq['Country'].value_counts()
country[0:21]
```
![1](https://user-images.githubusercontent.com/62857660/154602659-0f2849ab-c377-4af1-8a86-371668838e83.jpg)

# Ichimoku Cloud

Ichimoku is a technical indicator. The 5 lines of the cloud are:

- Tenkan Sen = Conversion Line: determine the direction of the short-term trend (yellow). Faster line.
- Kijun Sen = Base Line: avg for medium point and shows mid-term trend (red). Slower line.
- Senkou Span A = Leading Span A (green)
- Senkou Span B = Leading Span B (red)
- Chikou Line = Lagging Span (teal): helps to confirm signal.

**Few important concepts:**

- Formed between Span A and Span B, the cloud shows support and resistance.
- Span A and Span B are set 26 periods into the future.
- Chikou represents the closing price and set 26 periods in the past.
- Wider the cloud, the stronger the trend.

Try to not use the strategy for less than 1 hour.

**Reading the cloud:**

- Price is above the cloud: **UP** trend. Green color. Top of cloud is the support.
- Price is below the cloud: **DOWN** trend. Red color. Bottom of cloud is the resistant.
- Not recommended to trade when price is inside the cloud. Market is not trending. Use top of cloud as resistance and bottom as support.
- Tk/Golden Cross: when conversion past base from bottom to up, a **BUY** signal. If the price is above the cloud during this cross, it is a strong buy signal. If the price is below the cloud, you may want to wait until price is on top of the cloud. If the lagging span is crossing the price at the same time at the same direction, it's also another signal on buy. Set the stop loss at the narest local minimum.
- ![Screenshot 2022-02-16 004928](https://user-images.githubusercontent.com/62857660/154605360-e4baa24e-a6a0-43e3-b5e0-32c24dee65ec.jpg)
- Death Cross: when conversion past base from top to bottom, a **SELL** signal. If the price is below the cloud during this cross, it is a strong sell signal. If the price is above the cloud, you may want to wait until price is on bottom of the cloud before entering short positions. Set the stop loss at the narest local maximum.
- ![Screenshot 2022-02-16 005252](https://user-images.githubusercontent.com/62857660/154605375-95f9a898-1ddb-4baf-8411-c550f273e9f9.jpg)




Functions preparing the tickers and csv:
```
def get_column_from_csv(file, col_name):
    # Try to get the file and if it doesn't exist issue a warning
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print("File Doesn't Exist")
    else:
        return df[col_name]
        
tickers = get_column_from_csv("/content/Nasdaq.csv", "Symbol")
 
def save_to_csv_from_yahoo(folder, ticker):
    stock = yf.Ticker(ticker)
    
    try:
        print("Get Data for : ", ticker)
        # Get historical closing price data
        df = stock.history(period="5y")
    
        # Wait 2 seconds
        time.sleep(2)
        
        # Remove the period for saving the file name
        # Save data to a CSV file
        # File to save to 
        the_file = folder + ticker.replace(".", "_") + '.csv'
        print(the_file, " Saved")
        df.to_csv(the_file)
    except Exception as ex:
        print("Couldn't Get Data for :", ticker)
        
PATH = "/content/"
  
for x in range(0, 300):
    save_to_csv_from_yahoo(PATH, tickers[x])
    print("Finished")
    
def get_stock_df_from_csv(tickers):
  try:
    df = pd.read_csv(PATH + tickers + '.csv', index_col=0)
  except FileNotFoundError:
    print("File Doesn't Exist")
  else:
    return df
```

Testing the above code:

```get_stock_df_from_csv('COST')```
![1](https://user-images.githubusercontent.com/62857660/154604004-4a13e613-f18c-400b-8204-21aa847349c0.jpg)

Functions to add the bands and returns:
```
from os import listdir
from os.path import isfile, join

files = [x for x in listdir(PATH) if isfile(join(PATH, x))]
tickers = [os.path.splitext(x)[0] for x in files]

def add_daily_return_to_df(df):
  df['daily_return'] = (df['Close'] / df['Close'].shift(1)) - 1
  return df

def add_cum_return_to_df(df):
  df['cum_return'] = (1 + df['daily_return']).cumprod()
  return df

def add_bollinger_bands(df):
  df['middle_band'] = df['Close'].rolling(window=20).mean()
  df['upper_band'] = df['middle_band'] + 1.96 * df['Close'].rolling(window=20).std()
  df['lower_band'] = df['middle_band'] - 1.96 * df['Close'].rolling(window=20).std()
  return df
  
def add_ichimoku(df):
  #Conversion line
  hi_val = df['High'].rolling(window=9).max()
  lo_val = df['Low'].rolling(window=9).min()
  df['Conversion'] = (hi_val+lo_val)/2

  #Base Line
  hi_val2 = df['High'].rolling(window=26).max()
  lo_val2 = df['Low'].rolling(window=26).min()
  df['Baseline'] = (hi_val2+lo_val2)/2

  #Span A
  df['SpanA'] = ((df['Conversion'] + df['Baseline']) / 2)

  #Span B
  hi_val3 = df['High'].rolling(window=52).max()
  lo_val3 = df['Low'].rolling(window=52).min()
  df['SpanB'] = ((hi_val3 + lo_val3)/2).shift(26)

  #Laggine Span
  df['Lagging'] = df['Close'].shift(-26)
  return df

```
Getting the tickers
```
for x in tickers:
  try:
    print("Working on:", x)
    new_df = get_stock_df_from_csv(x)
    new_df = add_daily_return_to_df(new_df)
    new_df = add_cum_return_to_df(new_df)
    new_df = add_bollinger_bands(new_df)
    new_df = add_ichimoku(new_df)
    new_df.to_csv(PATH + x + '.csv')
  except Exception as ex:
    print(ex)
```
Preparing the plot abd bollinger band:
```
def plot_with_boll_bands(df, ticker):
  fig = go.Figure()

  candle = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick')

  upper_line = go.Scatter(x=df.index, y=df['upper_band'], 
                          line=dict(color='rgba(250,0,0,0.75)',
                          width=1), name='Upper Band')

  mid_line = go.Scatter(x=df.index, y=df['middle_band'], 
                          line=dict(color='rgba(0,0,250,0.75)',
                          width=1), name='Middle Band')
  
  lower_line = go.Scatter(x=df.index, y=df['lower_band'], 
                          line=dict(color='rgba(0,25,0,0.75)',
                          width=1), name='Lower Band')
  
  fig.add_trace(candle)
  fig.add_trace(upper_line)
  fig.add_trace(mid_line)
  fig.add_trace(lower_line)

  fig.update_xaxes(title='Date', rangeslider_visible=True)
  fig.update_yaxes(title='Price')

  fig.update_layout(title="Bollinger Bands", height=1200, width=1800, showlegend=True)

  fig.show()
  ```
  ```
  import plotly.graph_objs as go

fig = go.Figure()

candle = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick')

upper_line = go.Scatter(x=df.index, y=df['upper_band'], 
                          line=dict(color='rgba(250,0,0,0.75)',
                          width=1), name='Upper Band')

mid_line = go.Scatter(x=df.index, y=df['middle_band'], 
                          line=dict(color='rgba(0,0,250,0.75)',
                          width=1), name='Middle Band')
  
lower_line = go.Scatter(x=df.index, y=df['lower_band'], 
                          line=dict(color='rgba(0,25,0,0.75)',
                          width=1), name='Lower Band')
  
fig.add_trace(candle)
fig.add_trace(upper_line)
fig.add_trace(mid_line)
fig.add_trace(lower_line)

fig.update_xaxes(title='Date', rangeslider_visible=True)
fig.update_yaxes(title='Price')

fig.update_layout(title="Bollinger Bands", height=800, width=1200, showlegend=True)
```
Creating the cloud colors:
```
def get_fill_color(label):
    if label >= 1:
        return 'rgba(0,250,0,0.2)'
    else:
        return 'rgba(250,0,0,0.2)'
```
Putting it all together to plot the ichimoku cloud:
```
def get_ichimoku(df):
  candle = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick')

  df1 = df.copy()
  fig = go.Figure()
  df['label'] = np.where(df['SpanA'] > df['SpanB'], 1, 0) #return 1 for green, return 0 for red
  df['group'] = df['label'].ne(df['label'].shift()).cumsum()
  df = df.groupby('group')

  dfs = []
  for name, data in df:
    dfs.append(data)
  for df in dfs:
    fig.add_traces(go.Scatter(x=df.index, y=df.SpanA,
        line=dict(color='rgba(0,0,0,0)')))

    fig.add_traces(go.Scatter(x=df.index, y=df.SpanB,
        line=dict(color='rgba(0,0,0,0)'),
        fill='tonexty',
        fillcolor=get_fill_color(df['label'].iloc[0])))
    
  baseline = go.Scatter(x=df1.index, y=df1['Baseline'],
                        line=dict(color='red', width=3), name='Baseline')
  
  conversion = go.Scatter(x=df1.index, y=df1['Conversion'],
                          line=dict(color='gold', width=3), name='Conversion')
  
  lagging = go.Scatter(x=df1.index, y=df1['Lagging'],
                          line=dict(color='purple', width=2), name='Lagging')
  
  span_a = go.Scatter(x=df1.index, y=df1['SpanA'],
                          line=dict(color='green', width=2, dash='dot'), name='Span A')
  
  span_b = go.Scatter(x=df1.index, y=df1['SpanB'],
                          line=dict(color='red', width=2, dash='dot'), name='Span B')
  
  fig.add_trace(candle)
  fig.add_trace(baseline)
  fig.add_trace(conversion)
  fig.add_trace(lagging)
  fig.add_trace(span_a)
  fig.add_trace(span_b)

  fig.update_layout(height=800, width=1400, showlegend=True)
```
Testing the plot on a TSLA ticker:
```
ticker_wanted = get_stock_df_from_csv('TSLA')
get_ichimoku(ticker_wanted)
```
  














