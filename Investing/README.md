# Ichimoku Cloud and Optimal Portfolio

![flat-happy-man-with-golden-coin-head_88138-806](https://user-images.githubusercontent.com/62857660/154740772-247c0850-62e0-450d-bb9c-b5330216bfdf.jpg)

##### Author: Emi Ly

##### Date: Feb 18, 2022

##### [Tableau Dashboard]-Coming Soon
#

### INTRODUCTION

This project is not set out to recommend any stocks. I'm not a trader, not even close. However, it is my interest to learn more about how to read charts and using technical indicators. I learned the ichimoku code from Derek Banas, an Udemy instructor. He is an awesome coder and I will be continuing learning from his videos. I recommend anyone interested in the same topic not just copy the code. It helps to type out all the code, google/stackoverflow anything you don't understand, and most importantly, google the concept behind ichimoku cloud, moving average, etc. Once you have a more than a basic understanding, try to use the code to build your own portfolio and use the ichimoku cloud to test trades. I built upon Derek Banas original ichimoku code. I added the EDA, modified the ichimoku code and run it differently for a portfolio based on my preference. 

The code is seperated into 3 parts:
### 📊 [EDA on Nasdaq](#eda-on-nasdaq)
### 🌦 [Ichimoku Cloud](#ichimoku-cloud)
### 💯 [Finding an Efficient Portfolio](#finding-an-efficient-portfolio)

Dataset:
- From `import yfinance as yf` and https://www.nasdaq.com/market-activity/stocks/screener
- For the optimal portfolio, I only picked the top 300 based on market cap in descending order. If you have time, please run on all of the NASDAQ tickers. Alternatively, you can handpick your own list of tickers. 
- For the optimal portfolio, I used a 5 year timeframe. To enhance your analysis, please also run a 6 months and 1 year comparison.




## EDA on NASDAQ
Import the necessary libraries:
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

#For the sharpe ratio code later
risk_free_rate = 0.0125
```

Testing yfinance and saving to csv (manually picking the tickers):
```
tickers = ['AMZN', 'COST', 'ADBE', 'QQQ', 'NVDA', 'VUG', 'VGT', 'VOO', 'SPY', 'ARKK', 'ADBE','NVDA', 'AMD', 'MSFT', 'DIS', 'WMT', 'TGT', 'HD']

for ticker in tickers:
  stock = yf.Ticker(ticker)
  df = stock.history(period="5y")
  df.sort_index(axis = 0.,  ascending=False)
  df.to_csv(ticker + '.csv')
```

Read the Nasdaq.csv:
```
nasdaq = pd.read_csv("/content/Nasdaq.csv")
nasdaq.head(3)
```
![1](https://user-images.githubusercontent.com/62857660/154596067-179b2ee5-3483-4a30-89a6-eeb89ade275b.jpg)

Begin exploring the df with `nasdaq.info()`, `nasdaq.shape`, `nasdaq.dtypes`, `nasdaq.columns`.

Checking for missing values. IPO Year contains 40% of the missing data, which is alot! 
```
percent_missing = nasdaq.isnull().sum() * 100 / len(nasdaq)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df.sort_values(by=['percent_missing'], ascending=False)

#duplicated values
duplicated_value_df = nasdaq.duplicated(keep=False).value_counts(normalize=True) * 100
```
![4](https://user-images.githubusercontent.com/62857660/154598907-d79f8be5-03a1-45bf-b3a4-2a6f44179abb.jpg)


Let's explore IPO Year a little more. 2021 is the year with the most IPO. However, because 40% of data in the column the IPO Year is missing, so it's hard to say if 2021 is truly the year with the most IPO.
```
ipo_year = nasdaq['IPO Year'].value_counts().sort_values(ascending=False)
ax = ipo_year.plot(kind='barh', figsize=(25, 10), color='#86bf91', zorder=2, width=0.85)
```
![2](https://user-images.githubusercontent.com/62857660/154597871-ed2c2eb4-d7af-48db-9937-7af614eaff0f.png)

Using `nasdaq.loc[nasdaq['IPO Year'] == 2021]` to see some of the companies that IPO in 2021.
![3](https://user-images.githubusercontent.com/62857660/154598065-daaa00b8-3d69-4e4f-8678-c694c6ea3b20.jpg)


Using `nasdaq.describe()` for a summary of the numerical data. IPO Year is float64, then the many decimal places. It may be better to convert to a integer after you fix the NaN data. 

![9](https://user-images.githubusercontent.com/62857660/154600055-ac66b924-5848-476d-8a82-8851177b51ab.jpg)

Companis have the **earliest IPO year**:

```nasdaq.loc[nasdaq['IPO Year'] == 1972]```

![5](https://user-images.githubusercontent.com/62857660/154599781-b2d6adc9-4920-41a5-a380-cf9b67c02152.jpg)


Company has the **most market cap**: 

```nasdaq.loc[nasdaq['Market Cap'] == 3.000000e+12]```

![6](https://user-images.githubusercontent.com/62857660/154599785-2dd1e3d6-b5af-4a9a-b1f6-c34d719e502a.jpg)


Company has the **max net change**: 

```nasdaq.loc[nasdaq['Net Change'] == 1570]```

![7](https://user-images.githubusercontent.com/62857660/154599794-dffbe2bd-cbcb-4730-a13e-6754500d1f6e.jpg)

Company has the **least net change**:

```nasdaq.loc[nasdaq['Net Change'] == -86.16]```

![8](https://user-images.githubusercontent.com/62857660/154599796-fa937e84-74e3-4a0e-94d8-f610ba618767.jpg)

Make up of sectors:
```
print(nasdaq['Sector'].value_counts(normalize=True)*100
```
![2](https://user-images.githubusercontent.com/62857660/154600608-d19f8b30-a738-4e90-a97e-feb7beef571a.png)

Different industries in each sector: `nasdaq['Industry'].groupby(nasdaq['Sector']).value_counts()`
![3](https://user-images.githubusercontent.com/62857660/154601922-228a77c5-e008-488a-8e6c-10d3f5298780.jpg)

Top 20 represented countries: 
```
country = nasdaq['Country'].value_counts()
country[0:21]
```
![1](https://user-images.githubusercontent.com/62857660/154602659-0f2849ab-c377-4af1-8a86-371668838e83.jpg)

Bar plot representation. United States are most presented country, followed by China:
![3](https://user-images.githubusercontent.com/62857660/154611331-9c5c5bd1-4944-4cbb-896e-c8a4c1450bc8.png)


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
- Try to not use the strategy for less than 1 hour.
- 
**Reading the cloud:**

- Price is above the cloud: **UP** trend. Green color. Top of cloud is the support.
- Price is below the cloud: **DOWN** trend. Red color. Bottom of cloud is the resistant.
- Not recommended to trade when price is inside the cloud. Market is not trending. Use top of cloud as resistance and bottom as support.
- **Tk/Golden Cross:** when conversion past base from bottom to up, a **BUY** signal. If the price is above the cloud during this cross, it is a strong buy signal. If the price is below the cloud, you may want to wait until price is on top of the cloud. If the lagging span is crossing the price at the same time at the same direction, it's also another signal on buy. Set the stop loss at the narest local minimum.
- ![Screenshot 2022-02-16 004928](https://user-images.githubusercontent.com/62857660/154605360-e4baa24e-a6a0-43e3-b5e0-32c24dee65ec.jpg)
- **Death Cross:** when conversion past base from top to bottom, a **SELL** signal. If the price is below the cloud during this cross, it is a strong sell signal. If the price is above the cloud, you may want to wait until price is on bottom of the cloud before entering short positions. Set the stop loss at the narest local maximum.
- ![Screenshot 2022-02-16 005252](https://user-images.githubusercontent.com/62857660/154605375-95f9a898-1ddb-4baf-8411-c550f273e9f9.jpg)


**Source**:https://www.youtube.com/watch?v=PDPXvJ8W1G0


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

Testing the function on Costco stock:

```get_stock_df_from_csv('COST')```
![1](https://user-images.githubusercontent.com/62857660/154604004-4a13e613-f18c-400b-8204-21aa847349c0.jpg)

Function adding returns, bollinger bands and ichimoku lines:
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
![costco](https://user-images.githubusercontent.com/62857660/154733979-78763c2a-da4a-4259-a432-825dd72ca029.JPG)


Getting the tickers:
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

Function for the plotting:
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
  
  Plotting:
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
![1](https://user-images.githubusercontent.com/62857660/154610521-ebee1cbc-b4a6-4993-ab47-1c7a6d734dc6.jpg)





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
![cloud](https://user-images.githubusercontent.com/62857660/154622774-61b453c4-26fd-449e-8ab9-d1f3ede018ea.jpg)



# Finding an Efficient Portfolio

The attempt is to try the find a portfolio that have higher return and lower risk. We will see which tickers perform the best in cumulative return for the 5 years and analyze those tickers in each sector. You can pick out your own tickers from the sector analysis. For the portfolio, we will use the Sharpe Ratio to calculate the percentage of each ticker, check how many shares of each ticker's stock you will need to purchase and what is the total cost of investment. The price will be based on yesterday's closing price. 

What is **Cumulative Return**?
- The cumulative return is the total change in the investment price over a set time—an aggregate return, not an annualized one.
- Reinvesting the dividends or capital gains of an investment impacts its cumulative return.
- Cumulative return figures for ETFs and mutual funds typically omit the impact of annual expense ratios and other fees on the fund's performance.
- Taxes can also substantially reduce the cumulative returns for most investments unless they are held in tax-advantaged accounts.

What is the difference between Cumulative vs. Annualized Return?
- Annualized return is the return on investment received that year. 
- Cumulative return is the return on the investment in total.


Create a new df: `sec_df = pd.read_csv('/content/Nasdaq.csv')`. 
Because the ticker column is named 'Symbol', we will change to 'Ticker' to keep it consistent: 
`sec_df.rename(columns={'Symbol': 'Ticker'}, inplace=True)`. 

Here is our sectors:
![sector](https://user-images.githubusercontent.com/62857660/154612004-8e27ea47-9e63-4952-9cfd-1752b3fc2e87.jpg)

Create df for each sector:
```
finance_df = sec_df.loc[sec_df['Sector'] == "Finance"]
health_df = sec_df.loc[sec_df['Sector'] == "Health Care"]
tech_df = sec_df.loc[sec_df['Sector'] == "Technology"]
consumer_df = sec_df.loc[sec_df['Sector'] == "Consumer Services"]
goods_df = sec_df.loc[sec_df['Sector'] == "Capital Goods"]
nondurables_df = sec_df.loc[sec_df['Sector'] == "Consumer Non-Durables"]
energy_df = sec_df.loc[sec_df['Sector'] == "Energy"]
public_utilities_df = sec_df.loc[sec_df['Sector'] == "Public Utilities"]
industries_df = sec_df.loc[sec_df['Sector'] == "Basic Industries"]
misc_df = sec_df.loc[sec_df['Sector'] == "Miscellaneous"]
durables_df = sec_df.loc[sec_df['Sector'] == "Consumer Durables"]
transportation_df = sec_df.loc[sec_df['Sector'] == "Transportation"]
```

Calculate the cumulative return for each sector:
```
def get_cum_ret_for_stocks(stock_df):
    tickers = []
    cum_rets = []

    for index, row in stock_df.iterrows():
        df = get_stock_df_from_csv(row['Ticker'])
        if df is None:
            pass
        else:
            tickers.append(row['Ticker'])
            cum = df['cum_return'].iloc[-1]
            cum_rets.append(cum)
    return pd.DataFrame({'Ticker':tickers, 'CUM_RET':cum_rets})
  
finance_df = get_cum_ret_for_stocks(finance_df)
health_df = get_cum_ret_for_stocks(health_df)
tech_df = get_cum_ret_for_stocks(tech_df)
consumer_df = get_cum_ret_for_stocks(consumer_df)
goods_df = get_cum_ret_for_stocks(goods_df)
nondurables_df = get_cum_ret_for_stocks(nondurables_df)
energy_df = get_cum_ret_for_stocks(energy_df)
public_utilities_df = get_cum_ret_for_stocks(public_utilities_df)
industries_df = get_cum_ret_for_stocks(industries_df)
misc_df = get_cum_ret_for_stocks(misc_df)
durables_df = get_cum_ret_for_stocks(durables_df)
transportation_df = get_cum_ret_for_stocks(transportation_df)
```

Top 10 Cumulative Return Performance Per Sector. For example: information technology sector:
```
print('Tech:')
print(tech_df.sort_values(by=['CUM_RET'], ascending=False).head(10))
```
![tech](https://user-images.githubusercontent.com/62857660/154612237-aac7a18f-ec7f-484a-9f16-befb9c9228f3.jpg)


Top 10 for consumer services sector:
```
print('Consumer Services:')
print(consumer_df.sort_values(by=['CUM_RET'], ascending=False).head(10))
```
![consumerservices](https://user-images.githubusercontent.com/62857660/154612398-5e624329-88c1-41eb-83ab-4d868907c6ff.jpg)


Top 10 for health sector:
```
print('Health:')
print(health_df.sort_values(by=['CUM_RET'], ascending=False).head(10))
```
![health](https://user-images.githubusercontent.com/62857660/154612514-9fc4becb-6c23-49ce-a1df-6aa6c2f284c9.jpg)


Top 10 for capital goods sector:
```
print('Capital Goods:')
print(goods_df.sort_values(by=['CUM_RET'], ascending=False).head(10))
```
![capitalgoods](https://user-images.githubusercontent.com/62857660/154612848-4016a265-81a9-4e9d-90cc-327229ec837a.jpg)


For all the sectors combined, use `sec_cumret_df = get_cum_ret_for_stocks(sec_df)`


Getting the tickers to a list:
```
files = [x for x in listdir(PATH) if isfile(join(PATH, x))]
tickers = [os.path.splitext(x)[0] for x in files]
tickers
```

Create a df with the tickers from the saved csv:
```
def get_stock_df_from_csv(tickers):
  try:
    df = pd.read_csv(PATH + tickers + '.csv', index_col=0)
  except FileNotFoundError:
    print("File Doesn't Exist")
  else:
    return df

def merge_df_by_column_name(col_name, sdate, edate, *tickers):
  mult_df = pd.DataFrame()

  for x in tickers:
    df = get_stock_df_from_csv(x)
    mask = (df.index >= sdate) & (df.index <=edate)
    mult_df[x] = df.loc[mask][col_name]
  
  return mult_df
```

Creating a portfolio list. As the introduction stated, this project is not set out to recommend any stocks. I'm not a trader, not even close. Ideally, you should pick out some tickers from each sector and create an efficient portofolio. You can read more about the **Modern Portfolio Theory** at:https://www.investopedia.com/terms/m/modernportfoliotheory.asp

- The modern portfolio theory (MPT) is a method that can be used by risk-averse investors to construct diversified portfolios that maximize their returns without unacceptable levels of risk.
- The modern portfolio theory can be useful to investors trying to construct efficient and diversified portfolios using ETFs.
- Investors who are more concerned with downside risk might prefer the post-modern portfolio theory (PMPT) to MPT.
- The modern portfolio theory (MPT) was a breakthrough in personal investing. It suggests that a conservative investor can do better by choosing a mix of low-risk and riskier investments than by going entirely with low-risk choices. More importantly, it suggests that the more rewarding option does not add additional overall risk. This is the key attribute of portfolio diversification.
- The post-modern portfolio theory (PMPT) does not contradict these basic assumptions. However, it changes the formula for evaluating risk in an investment in order to correct what its developers perceived as flaws in the original. Followers of both theories use software that relies on either MPT or PMPT to build portfolios that match the level of risk that they seek.

**In this notebook, I just picked the top 20 based on cumulative return of the last 5 year.** The top 20 may be different if you run your analysis on a 6 month, 1 year, 26 weeks, ettc etc. You can also handpick your top 5, 10, 20, however many you like. 

```
print('ALL sectors:')
top_20 = sec_cumret_df.sort_values(by=['CUM_RET'], ascending=False).head(20)
top_20
```
![top20](https://user-images.githubusercontent.com/62857660/154615520-fe1f7340-0521-4df0-92ff-a679cf31676e.jpg)


Price changes from the top 20 tickers over the 5 years:
```
mult_df = merge_df_by_column_name('Close', S_DATE, E_DATE, *port_list)
mult_df
```
![plot0](https://user-images.githubusercontent.com/62857660/154617068-039c7498-7a3c-4168-900c-835d6496c918.jpg)


Plotting the price changes:
```
import plotly.express as px

fig = px.line(mult_df, x=mult_df.index, y=mult_df.columns)
fig.update_xaxes(title="Date", rangeslider_visible=True)
fig.update_yaxes(title="Price")
fig.update_layout(height=900, width=1500, 
                  showlegend=True)
fig.show()
```
![plot](https://user-images.githubusercontent.com/62857660/154617079-ecaec72d-68c7-4ccb-b6de-d0ab0cd5edf9.jpg)


Obtain the annualized return for a 252 trading days for the 5 years:
```
returns = np.log(mult_df / mult_df.shift(1))
mean_ret = returns.mean()*100 * 252  #252 trading days
mean_ret
```
![mean_return](https://user-images.githubusercontent.com/62857660/154617222-e9ed281b-591f-4673-92ff-e142e441d383.jpg)



For an efficient portfolio under MPT, pick a portfolio combination with correlaton **less than 0.50**:
```
corr = returns.corr()

# Correlation heatmap
import seaborn as sns
fig, ax = plt.subplots(figsize=(15, 12))
mask = np.triu(np.ones_like(corr, dtype=np.bool))

mask = mask[1:, :-1]
corr = corr.iloc[1:,:-1].copy()

cmap = sns.color_palette("hls", 8)

sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", 
           linewidths=5, cmap=cmap, vmin=-1, vmax=1, 
           cbar_kws={"shrink": .8}, square=True)

yticks = [i.upper() for i in corr.index]
xticks = [i.upper() for i in corr.columns]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
plt.xticks(plt.xticks()[0], labels=xticks)

plt.show()
```
![download](https://user-images.githubusercontent.com/62857660/154622274-9c9e43fb-1f43-4bb6-bf3d-c6463fe9ac25.png)




Return and Risk of 10000 Combinations:
```
p_ret = []   #return list
p_vol = []   #volatility risk, std from mean
p_SR = []
p_wt = []     #amt of each stock we have

for x in range(10000):
  p_weights = np.random.random(num_stocks)
  p_weights /= np.sum(p_weights)

  ret_1 = np.sum(p_weights * returns.mean()) * 252
  p_ret.append(ret_1)

  vol_1 = np.sqrt(np.dot(p_weights.T, np.dot(returns.cov() * 252, p_weights)))
  p_vol.append(vol_1)

  SR_1 = (ret_1 - risk_free_rate) / vol_1
  p_SR.append(SR_1)

  p_wt.append(p_weights)

p_ret = np.array(p_ret)
p_vol = np.array(p_vol)
p_SR = np.array(p_SR)
p_wt = np.array(p_wt)

p_ret, p_vol, p_SR, p_wt
```
```
ports = pd.DataFrame({'Return': p_ret, 'Volatility': p_vol})
print(ports)
ports.plot(x='Volatility', y='Return', kind='scatter', figsize=(30,15))
```
Y axis is return and X axis is risk. Pick a combo that's lower risk and higher return:
![download (1)](https://user-images.githubusercontent.com/62857660/154617545-ec8c0d15-7312-44d7-906b-fe2cdce20e31.png)

We will be using the **Sharpe Ratio**. The Sharpe ratio was developed by Nobel laureate William F. Sharpe and is used to help investors understand the return of an investment compared to its risk.

- The Sharpe ratio adjusts a portfolio’s past performance—or expected future performance—for the excess risk that was taken by the investor.
- A high Sharpe ratio is good when compared to similar portfolios or funds with lower returns.
- The Sharpe ratio has several weaknesses, including an assumption that investment returns are normally distributed.
- Read more here for a example where you will use the Sharpe Ratio: https://www.investopedia.com/terms/s/sharperatio.asp

Calculate the Sharpe Ratio:
```
SR_idx = np.argmax(p_SR)

i = 0

while i < num_stocks: 
  print('Stock : %s : %2.2f' % (port_list[i],
                                (p_wt[SR_idx][i] * 100)))
  i+=1

print('\nVolatility:', p_vol[SR_idx])
print("Return: ", p_ret[SR_idx])
```


Getting the shares needed and shares cost. We will need to obtain the closing price with the lowest Sharpe ratio: 
```
def get_port_shares(one_price, force_one, wts, prices):
  num_stocks = len(wts)
  shares = []

  cost_shares = []

  i = 0
  while i < num_stocks: 
    max_price = one_price * wts[i]
    num_shares = int(max_price / prices[i])
    if(force_one & (num_shares == 0)):
      num_shares = 1
    shares.append(num_shares)
    cost = num_shares * prices[i]
    cost_shares.append(cost)
    i += 1
  return shares, cost_shares
  
def get_port_weighting(share_cost):
    
    # Holds weights for stocks
    stock_wts = []
    # All values summed
    tot_val = sum(share_cost)
    print("Total Investment :", tot_val)
    
    for x in share_cost:
        stock_wts.append(x / tot_val)
    return stock_wts
 
def get_port_val_by_date(date, shares, tickers):
    port_prices = merge_df_by_column_name('Close',  date, 
                                  date, *port_list)
    # Convert from dataframe to Python list
    port_prices = port_prices.values.tolist()
    # Trick that converts a list of lists into a single list
    port_prices = sum(port_prices, [])
    return port_prices
    
#Convert the weights to percentage
port_wts = p_wt[SR_idx].tolist()
port_wts = [i*100 for i in port_wts]
```

In the final output, we will need to find the price of the stock with the minimal weight (or weight closest to 1). This is the one_price parameter:
```
get_port_shares(one_price, force_one, wts, prices)

def get_price_at_min_weight(ticker_list,weights,price):
  df = pd.DataFrame({'Ticker': ticker_list,
                   'Weights': weights,
                   'Price': price})

  min = df[df['Weights'] == df['Weights'].min()]
  min_price = min['Price']
  return(min_price)

min_price = get_price_at_min_weight(port_list, port_wts, port_prices)
```

Getting the previous trading date and convert back to string to use in the final output function:
```
from datetime import timedelta
from datetime import datetime
 
# Get today's date
today = datetime.today()
 
# Yesterday date
yesterday = today - timedelta(days = 1)
yesterday = yesterday.strftime('%Y-%m-%d')
type(yesterday)
yesterday
```

Final output:
```
# Get all stock prices on the starting date
port_df_start = merge_df_by_column_name('Close',  yesterday, 
                                  yesterday, *port_list)

# Convert from dataframe to Python list
port_prices = port_df_start.values.tolist()

# Trick that converts a list of lists into a single list
port_prices = sum(port_prices, [])

tot_shares, share_cost = get_port_shares(min_price, True, port_wts, port_prices)
print("Shares :", tot_shares)
print("Share Cost :", share_cost)

# Get list of weights for stocks
stock_wts = get_port_weighting(share_cost)
print("Stock Weights :", stock_wts)

# Get value at end of year
get_port_val_by_date(E_DATE, tot_shares, port_list)
```
![final](https://user-images.githubusercontent.com/62857660/154621351-94f3e7b5-4c0f-410f-a94c-8e62eb29a5f0.jpg)



## 🌟Summary 🌟

Closing price of the tickers in the portfolio list:`port_df_start`
![list](https://user-images.githubusercontent.com/62857660/154621770-1ea6a3bc-ed75-4e01-ab74-a1f65b9e8734.jpg)


Return and volatility:
```
print('Volatility:', p_vol[SR_idx]*100)
print('Return: ', p_ret[SR_idx]*100)
print('Total Share Needed: ', sum(tot_shares))
print('Total Share Cost: ', sum(share_cost))
```
![return](https://user-images.githubusercontent.com/62857660/154621853-947145cd-b9d7-4306-8166-7b8b3ffe4801.jpg)

Summary dataframe:
```
df = pd.DataFrame({'Ticker': port_list,
                   'Weights': port_wts,
                   'Shares': tot_shares,
                   'Closing Price': port_prices,
                   'Share Cost': share_cost,
                   'Cumulative Return': top_10['CUM_RET'],
                   'Mean Return': mean_ret.tolist()})
```
![df](https://user-images.githubusercontent.com/62857660/154621941-ee551af9-c7c9-466d-9aaf-65d86c973838.jpg)



![happy](https://user-images.githubusercontent.com/62857660/154741070-e4e08e6e-5f17-482f-9e64-9276e79687bb.jpg)


Image Source: https://www.freepik.com/free-vector/happy-rich-people-growing-financial-plant-watering-money-tree-pile-cash-analyzing-wealth-prosperity_13146550.htm#query=happy%20investor&position=31&from_view=search and https://www.freepik.com/free-vector/flat-happy-man-with-golden-coin-head_22654320.htm#query=happy%20investor&position=18&from_view=search
