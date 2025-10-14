# === IMPORTS ===
# CHANGED: import Spot instead of Client
from binance.spot import Spot
import pandas as pd
import pandas_ta as ta
import datetime as dt
import time

# === 1. Initialize API client ===
# You can use an API key if you have one, or leave them blank for public dat
key_name = "cs230_project"
api_key = '4xBdatBvCG4RIJKKjwFJy5HVVV9CM4TamudtrWTETWaA2Vomeb1dRcyADoTbtqd9'
api_secret = 'LWxNl18AJpRrRy4o23jxJnB556FMikyMOHegYQp9PCF7kU69qVzsVWB3ojNg95LP'

# CHANGED: Initialize Spot client with base_url for Binance.US
client = Spot(api_key=api_key, api_secret=api_secret, base_url='https://api.binance.us')

# === 2. Set parameters ===
symbol = 'BTCUSDT'
interval = '30m'  # CHANGED: Spot client uses strings, not constants
end_time = dt.datetime.now() #Ocotber 12, 2025 @ 5:00 PM PST
start_time = end_time - dt.timedelta(days=365 * 10)  # 10 years ago

#Fix end_time as the current time.
# Progressively march start time forward by 1000 points until start time >= end time, then stop.

klines = []
num_calls = 0                                                                         #keep track of the number of API calls made
while True:
    # Fetch candles
    # No matter how wide the time range you specify with start_str and end_str,
    # Binance will only send you the earliest 1000 candles that fit that interval.

    # CHANGED: use client.klines() instead of get_historical_klines()
    #for attempt in range(5):
        #try:
    new_klines = client.klines(
        symbol=symbol,
        interval=interval,
        startTime=int(start_time.timestamp() * 1000),
        endTime=int(end_time.timestamp() * 1000),
        limit=1000
    )
        #except requests.exceptions.ConnectionError as e:
        #    print(f"Connection error (attempt {attempt+1}/5): {e}")
        #    time.sleep(3)
        #raise RuntimeError("Failed to reconnect after 5 attempts.")

    

    print(f"API Call #{num_calls}: Fetched {len(new_klines)} candles starting from {start_time}")
    
    if not new_klines:                                                                 # Stop if no more data is returned
        break
    
    klines.extend(new_klines)                                                          # Append new candles to the list                        
    
    # Update start_time to the last returned candle + 30 minutes
    last_open_time = new_klines[-1][0] / 1000                                          # Convert ms to seconds
    start_time = dt.datetime.fromtimestamp(last_open_time) + dt.timedelta(minutes=30)
    
    if start_time >= end_time:                                                         # Stop when we reach the end
        break

    num_calls += 1
    time.sleep(0.25)                                                                   # Sleep to avoid hitting rate limits

# === 3. Convert to DataFrame ===
df = pd.DataFrame(klines, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore'
])

# === 4. Format the data ===
df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)


# ---- 5. Add technical indicators ---- #
RSI_length = 14
bbands_length = 20
max_length = max(RSI_length, bbands_length) + 1                                         # +1 to ensure enough data for indicators   

df['RSI_14'] = ta.rsi(df['close'], length=RSI_length)                                   # RSI with 14-period                                     # 50-period Simple Moving Average
bbands = ta.bbands(df['close'], length=bbands_length, std=2)                            # Returns a DataFrame with columns 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0'
df = pd.concat([df, bbands], axis=1)




# ---- 6. Resize DataFrame to remove initial NaN values from indicators ---- #
df = df.iloc[max_length:].reset_index(drop=True)
print(f"DataFrame resized to {len(df)} rows after removing initial NaN values from indicators.")
print(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")                          

df_without_NaNs = df.copy()  # Keep a copy without NaNs for reference
NaNrows_df = df[df.isna().any(axis=1)]  # DataFrame of rows with NaNs
print(f"Number of rows with NaN values: {len(NaNrows_df)}")

df.dropna(inplace=True)  # Remove rows with NaN values
df.drop(columns=['open_time', 'quote_asset_volume', 'taker_buy_base', 'taker_buy_quote', 'ignore'], inplace=True)
#df.rename(columns={"close_time": "close_time_UNIX"})


# === 5. Save to CSV ===
df.to_csv('BTCUSDT_30m_10years.csv', index=False)
print(f"Saved {len(df)} rows to BTCUSDT_30m_10years.csv")
