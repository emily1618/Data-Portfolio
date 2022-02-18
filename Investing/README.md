# Ichimoku Cloud and Optimal Portfolio
##### Author: Emi Ly

##### Date: Feb 18, 2022

##### [Tableau Dashboard]-Coming Soon
#

### INTRODUCTION

This project is not set out to recommend any stocks. I'm not a trader, not even close. However, it is my interest to learn more about how to read charts and using technical indicators. I learned the ichimoku code from Derek Banas, an Udemy instructor. He is an awesome coder and I will be continuing learning from his videos. I recommend anyone interested in the same topic NOT to just copy the code. It helps to type out all the code, google/stackoverflow anything you don't understand, and most importantly, google the concept behind ichimoku cloud, moving average, etc. Once you have a more than a basic understanding, try to use the code to build your own portfolio and use the ichimoku cloud to test trades. I only scratched the surface and there are so much more to learn! 

Dataset:
- From `import yfinance as yf` and https://www.nasdaq.com/market-activity/stocks/screener
- I only picked the top 300 based on market cap. If you have time, please run on all of the NASDAQ tickers!
- I used a 5 year timeframe. To enhance your analysis, please also run a 6 months and 1 year comparison.

The code is seperated into 3 parts:
### 📊 [EDA on NYSE and NASDAQ](#eda-on-nyse-and-nasdaq)
### 🌦 [Ichimoku Cloud](#ichimoku-cloud)
### 🧗‍ [Finding a Optimal Portfolio](#finding-a-optimal-portfolio)

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

Downloading the data from yfinance and saving to csv:
```
for ticker in tickers:
  stock = yf.Ticker(ticker)
  df = stock.history(period="5y")
  df.sort_index(axis = 0.,  ascending=False)
  df.to_csv(ticker + '.csv')
```


