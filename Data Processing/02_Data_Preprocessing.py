# Data preprocessing file:

import numpy as np
import pandas as pd
import pandas_ta as ta
import sys
import matplotlib.pyplot as plt
import pywt
from scipy.signal import lfilter, butter
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

### Parameters ###
SEED = 42
rng = np.random.default_rng(seed=SEED)

#### Can be any combination of True/False ####
COMBINE_DATA = False
BLOCK_SHUFFLE = False 
TREND_WEIGHT = 3
DENOISE = True
SMOOTH_FACTOR = 100    #observation noise is N times larger than process noise. for denoising


BLOCK_SIZE = 2048
HORIZON = 2
EPS = 1e-8
TA_LENGTH = 14 #technical indicator window for TA library functions
SELECT_RANGE = False #used if you want to select a later range of the data, rather than all of it
USE_MACD = True
SEQ_LEN = 32
train_frac, val_frac, test_frac = 0.9, 0.05, 0.05
batch_size = 256  # must be power of 2 for efficient training
#############################################




#### WAVELET DENOISING ####
def swt_denoise_multifeature(x, wavelet='db4', level=3, thresh_scale=0.7):
    m = 2 ** level
    target = int(np.ceil(len(x) / m) * m)
    pad_len = target - len(x)

    # pad on the right using last value
    if pad_len > 0:
        x_pad = np.pad(x, (0, pad_len), mode='edge')
    else:
        x_pad = x

    coeffs = pywt.swt(x_pad, wavelet=wavelet, level=level)
    d1 = coeffs[-1][1]
    sigma = np.median(np.abs(d1)) / 0.6745
    lam = thresh_scale * sigma * np.sqrt(2 * np.log(len(x_pad)))
    coeffs_t = [(cA, pywt.threshold(cD, lam, mode='soft')) for (cA, cD) in coeffs]
    x_rec = pywt.iswt(coeffs_t, wavelet)
    x_rec = x_rec.astype(np.float32)
    return x_rec[:len(x)]  

def dwt_denoise_multifeature_fast(
    X, wavelet='db4', level=3, thresh_scale=0.7,
    soft_or_hard='soft', mode='smooth', eps=1e-8
):
    num_seq, seq_len, num_feat = X.shape
    out = np.empty_like(X, dtype=np.float32)

    for f in range(num_feat):
        for s in range(num_seq):
            x = X[s, :, f]
            coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode=mode)
            d1 = coeffs[-1]
            sigma = np.median(np.abs(d1)) / 0.6745 + eps
            lam = thresh_scale * sigma * np.sqrt(2 * np.log(len(x)))
            coeffs_thr = [coeffs[0]] + [
                pywt.threshold(c, lam, mode=soft_or_hard) for c in coeffs[1:]
            ]
            x_rec = pywt.waverec(coeffs_thr, wavelet=wavelet, mode=mode)
            if len(x_rec) != len(x):
                x_rec = x_rec[:len(x)] if len(x_rec) > len(x) else np.pad(x_rec, (0, len(x) - len(x_rec)), 'edge')
            out[s, :, f] = np.clip(x_rec, eps, None).astype(np.float32)

    return out

def kalman_denoise_multifeature(
    X,
    process_var=1e-4,     # process noise variance (Q)
    smoothing_amt=SMOOTH_FACTOR     # multiplies with obs_var (R)
   
):
    

    """
    Apply a simple 1D Kalman filter to each feature column independently.
    State model: x_t = x_{t-1} + w_t
    Observation: z_t = x_t + v_t

    Parameters
    ----------
    X : np.ndarray
        Shape (n_timesteps, n_features)
    process_var : float
        Process noise variance (how much the hidden state can drift)
    obs_var : float
        Observation noise variance (how noisy measurements are)

    Returns
    -------
    np.ndarray
        Smoothed data of same shape.
    """

    obs_var = process_var * smoothing_amt  # observation noise variance (R)


    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1) #turns (X,) into (X,1)
    n, m = X.shape
    out = np.zeros_like(X)

    for j in range(m):
        z = X[:, j]
        x_est = np.zeros(n, dtype=np.float32)
        P = 1.0  # initial estimate uncertainty
        x_est[0] = z[0]

        for t in range(1, n):
            # Prediction
            x_pred = x_est[t-1]
            P_pred = P + process_var

            # Kalman Gain
            K = P_pred / (P_pred + obs_var)

            # Update
            x_est[t] = x_pred + K * (z[t] - x_pred)
            P = (1 - K) * P_pred

        out[:, j] = x_est

    return out


"""
 - DO NOT denoise volume or num trades - spikes are signal!!
 - DO NOT denoise technical indicators - they are already pretty smooth
"""

#### Plotting Function ####
def plot_features(df, cols_to_plot, window_size=1000, title_suffix=''):
    """Plot specified columns from dataframe in a random window."""
    start_idx = np.random.randint(0, len(df) - window_size)
    window_slice = slice(start_idx, start_idx + window_size)
    plt.figure(figsize=(14, 8))
    for col in cols_to_plot:
        plt.plot(df[col].iloc[window_slice].values, label=col, linewidth=1.5, alpha=0.8)
    plt.title(f'{title_suffix} (Window: {window_slice.start}-{window_slice.stop})', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

#### Initial Data Processing ####
def load_data(fname):
    df = pd.read_csv(fname)
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    if SELECT_RANGE:
        START_DATE = pd.Timestamp('2022-01-01')  # change if you want a different cutoff
        df = (
            df[df['close_time'] >= START_DATE]
            .sort_values('close_time')
            .reset_index(drop=True)
        )
    
    print(f"Loaded {len(df)} rows from CSV")
    print(f"Date range: {df['close_time'].min().strftime('%m/%d/%Y')} to {df['close_time'].max().strftime('%m/%d/%Y')}")  #print data range in MM/DD/YYYY format
    
    return df


#### GOOGLE TRENDS ####
def fetch_bitcoin_trends(L):
    """
    Fetch daily Google Trends data for 'Bitcoin' and expand to hourly frequency.

    Args:
        L: Length of hourly dataframe to match

    Returns:
        pd.Series of length L with piecewise constant daily values repeated hourly
    """
    pytrends = TrendReq(hl='en-US', tz=360)

    # Calculate how many days we need
    n_days = L // 24

    # Fetch trends data (Google Trends max is ~270 days for daily data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)

    pytrends.build_payload(['Bitcoin'], timeframe=f'{start_date.strftime("%Y-%m-%d")} {end_date.strftime("%Y-%m-%d")}')
    trends_df = pytrends.interest_over_time()

    # Get Bitcoin column and take first n_days
    daily_values = trends_df['Bitcoin'].values[:n_days]

    # Repeat each daily value 24 times for hourly frequency
    hourly_values = np.repeat(daily_values, 24)

    # Handle remaining samples by repeating the last value
    remainder = L - len(hourly_values)
    if remainder > 0:
        hourly_values = np.concatenate([hourly_values, np.repeat(daily_values[-1], remainder)])

    return pd.Series(hourly_values[:L])



### funcs I added ################
def zigzag(series, percent=0.02):
    """
    TradingView-style ZigZag with linear interpolation between pivot points.
    percent = reversal threshold (0.02 = 2%)
    """
    s = series.values
    idx = series.index
    n = len(s)

    if n < 3:
        return series.copy()

    pivots = np.full(n, np.nan)
    last_pivot = 0
    last_pivot_price = s[0]
    direction = 0  # +1 uptrend, -1 downtrend

    for i in range(1, n):
        change = (s[i] - last_pivot_price) / last_pivot_price

        if direction == 0:
            if change > percent:
                direction = +1
                last_pivot = i
                last_pivot_price = s[i]
                pivots[i] = s[i]
            elif change < -percent:
                direction = -1
                last_pivot = i
                last_pivot_price = s[i]
                pivots[i] = s[i]

        elif direction == +1:
            if s[i] > last_pivot_price:   # new high continues trend
                last_pivot = i
                last_pivot_price = s[i]
            elif change < -percent:       # down reversal
                direction = -1
                last_pivot = i
                last_pivot_price = s[i]
                pivots[i] = s[i]

        elif direction == -1:
            if s[i] < last_pivot_price:   # new low continues trend
                last_pivot = i
                last_pivot_price = s[i]
            elif change > percent:        # up reversal
                direction = +1
                last_pivot = i
                last_pivot_price = s[i]
                pivots[i] = s[i]

    # ---- NOW PERFORM LINEAR INTERPOLATION BETWEEN PIVOTS ----
    zigzag = pd.Series(np.nan, index=idx)

    # get pivot indices
    pivot_idx = np.where(~np.isnan(pivots))[0]

    if len(pivot_idx) < 2:
        # not enough pivots -> no zigzag possible
        return series.copy()

    for j in range(len(pivot_idx) - 1):
        a = pivot_idx[j]
        b = pivot_idx[j + 1]

        y0 = pivots[a]
        y1 = pivots[b]

        # linear interpolation from pivot a to pivot b
        steps = b - a
        if steps <= 0:
            continue

        line = np.linspace(y0, y1, steps + 1)
        zigzag.iloc[a:b+1] = line

    # Fill start/end if needed
    zigzag.iloc[:pivot_idx[0]] = pivots[pivot_idx[0]]
    zigzag.iloc[pivot_idx[-1]:] = pivots[pivot_idx[-1]]

    return zigzag

def feature_engineer2(df):
    #trends = fetch_bitcoin_trends(df.shape[0])

    print("\n\nBefore Processing: df.shape")

    df["close"] = kalman_denoise_multifeature(df["close"])
    df["zigzag"] = zigzag(df["close"])
    df = df.ffill() #get rid of NaNs

    scaler = StandardScaler() 
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    
    plot_features(df_scaled, cols_to_plot=df.columns.to_list())

    w_close, w_vol, w_RSI, w_zig, w_trends = 1, 1, 1, 2, 2
    weights = pd.Series([w_close, w_vol, w_RSI, w_zig, w_trends], index=df.columns)
    df_weighted = df * weights
    # UNCOMMENT if loading in original BTCUSDT_30m_10years.csv dataset
    #df = df.drop(columns=["open","high","low","number_of_trades","BBL_20_2.0_2.0","BBM_20_2.0_2.0","BBU_20_2.0_2.0","BBB_20_2.0_2.0","BBP_20_2.0_2.0"])
    
    
    df.to_csv("new_features_df.csv")

    # Compute log returns: all OHLC relative to previous close price
    df[["close", "RSI_14", "zigzag"]] = np.log(df[["close", "RSI_14", "zigzag"]] + 1e-8).diff().fillna(0)
   

    ### REMOVE UN-NEEDED COLUMNS ###
    close_times = df.index
    close_times.to_frame().to_csv('datapoint_timestamps.csv', index=True, header=True)
    close_idx = df.columns.get_loc('close')

    # Convert datetime to milliseconds for fast numpy operations
    timestamps = close_times.astype(np.int64) // 10**6  # Convert to milliseconds


    
    print("\n\nAfter Processing")
    print(df.head())

    return df.to_numpy(), df.columns.tolist(), close_idx, timestamps
################



def feature_engineer1(df):
    # Bollander Band Width and Position
    df['BB_width'] =  (df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0']) / (df['BBM_20_2.0_2.0'] + 1e-8)
    df['BB_position'] = (df['close'] - df['BBL_20_2.0_2.0']) / (df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0'] + 1e-8)

    #MACD 
    if USE_MACD:
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        df.drop(columns=[ 'MACD_12_26_9', 'MACDs_12_26_9'], inplace=True)

    #Standard Moving Average
    df["SMA_14"] = ta.sma(df["close"], length=14)  
    df = df.dropna(axis=0, how='any').reset_index(drop=True)

    # Taking returns centers the timeseries around 1. Taking log returns centers around 0 and improves stability of training.
    if DENOISE:
        OHLC = df.iloc[:, :4].to_numpy()
        original_close = df.iloc[:, 3].copy()  # Save original for comparison
        denoised_OHLC = kalman_denoise_multifeature(OHLC, smoothing_amt=SMOOTH_FACTOR)

        n_original = len(df)
        n_denoised = len(denoised_OHLC)
        if n_denoised < n_original:
            trim = n_original - n_denoised
            df = df.iloc[trim:].reset_index(drop=True)
            original_close = original_close.iloc[trim:].reset_index(drop=True)
            denoised_OHLC = denoised_OHLC[-len(df):,:]  # trim equally from denoised array

        # Replace the first 4 columns with the denoised values
        print(df.shape, denoised_OHLC.shape)
        df.iloc[:, :4] = denoised_OHLC

        # Visualize denoising effect on close price (random 1000-point window)
    print(df.columns)
    plot_features(df, cols_to_plot=["close"])

    # # Check for negative values in the first 4 columns
    neg_mask = (df.iloc[:, :4] < 0)
    print(f"Number of negative values: {neg_mask.sum().sum()}")   # Should print 0 if no negatives exist

    EPS = 1e-8
    # Compute log returns: all OHLC relative to previous close price
    prev_close = df.iloc[:, 3].shift(1)  # Previous close (4th column, index 3)
    log_returns = np.log(np.clip(df.iloc[:, :4], EPS, None).div(np.clip(prev_close, EPS, None), axis=0))
    log_returns = log_returns.fillna(0)
    df.iloc[:,:4] = log_returns
    df.columns = ['log_return_open', 'log_return_high', 'log_return_low', 'log_return_close'] + list(df.columns[4:])



    ### REMOVE UN-NEEDED COLUMNS ###
    close_times = df['close_time']
    close_times.to_csv('datapoint_timestamps.csv', index=True, header=True)

    # Keep timestamps as numpy array for sequencing
    timestamps = close_times.to_numpy()

    df.drop(columns=['close_time', 'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0', 'BBB_20_2.0_2.0', 'BBP_20_2.0_2.0', 'number_of_trades'], inplace=True)

    if USE_MACD:
        df = df.iloc[33:, :]  # ONLY use FOR MACD
        timestamps = timestamps[33:]  # Keep timestamps aligned

    df.columns.to_series().to_csv("column_names.csv", index=False, header=False)
    arr = df.to_numpy()
    close_idx = df.columns.get_loc('log_return_close')
    print(f"Columns After Processing: {df.columns.tolist()}\n\n")

    return arr, df.columns.tolist(), close_idx, timestamps


#### Functions for Shuffling ####
def block(arr, BLOCK_SIZE):
    """
    Accepts: (num_datapoints, num_features)-size numpy array and 
    Returns: (num_blocks, block_size, num_features)-size numpy array, 
    
    Where num_blocks = floor(num_datapoints/block_size), and the remaining earlier datapoints (the earlier indices) are dropped
    so that each block is the same size
    """
    num_datapoints, num_features = arr.shape
    num_blocks = num_datapoints // BLOCK_SIZE  # floor division

    # Drop the earlier datapoints so that we can evenly reshape
    trimmed_arr = arr[-num_blocks * BLOCK_SIZE:, :]

    # Reshape into blocks
    blocked_data = trimmed_arr.reshape(num_blocks, BLOCK_SIZE, num_features)

    return blocked_data

def shuffle_blocks(X, Y):
    perm = rng.permutation(len(X)) #ensures both get shuffled in the same way
    return X[perm], Y[perm]

def block_and_sequence(arr, BLOCK_SIZE, close_idx):
    blocked_data = block(arr, BLOCK_SIZE=BLOCK_SIZE)
    num_blocks, block_size, num_features = blocked_data.shape
    num_sequences = block_size - SEQ_LEN - HORIZON
    if num_sequences <= 0:
        raise ValueError(f"BLOCK_SIZE ({BLOCK_SIZE}) must be > SEQ_LEN+HORIZON ({SEQ_LEN+HORIZON})")

    blocked_Xs = np.zeros((num_blocks, num_sequences, SEQ_LEN, num_features)) #size (num_blocks, num_sequences, sequence_len, num_features)  
    blocked_Ys = np.zeros((num_blocks, num_sequences, 1)) #size (num_blocks, num_sequences, 1) 

    for i in range(num_blocks):
        block_i = blocked_data[i, :, :]
        block_X, block_Y = sequence(block_i, SEQ_LEN, HORIZON, close_idx)
        blocked_Xs[i] = block_X
        blocked_Ys[i] = block_Y

    # last_block_idx = -1
    # X_last_block = blocked_Xs[last_block_idx]
    # Y_last_block = blocked_Ys[last_block_idx]

    # # Save this block for reconstruction (temporally latest, still ordered)
    # np.savez_compressed("last_temporal_block.npz",
    #     X_block=X_last_block,
    #     Y_block=Y_last_block
    # )
    # print(f"Saved temporally last block for reconstruction with shape: {X_last_block.shape}")

    return blocked_Xs, blocked_Ys



#### Critical Steps: Sequence, Split, Normalize ####
def sequence(data, window, HORIZON, close_idx, timestamps=None):
    """
    Create sequences with optional timestamp tracking.

    For each sequence:
    - X spans timestamps[i : i+window]
    - Y corresponds to timestamp[i+window+HORIZON-1] (the target prediction time)
    """
    T = data.shape[0]
    starts = np.arange(0, T - window - HORIZON)

    # --- Build feature sequences (X) ---
    X = np.stack([data[i:i+window] for i in starts], axis=0)
    # Y is HORIZON steps ahead of the last element in X
    # For sequence starting at i: X ends at i+window-1, Y is at i+window+HORIZON-1
    # Using array slicing: Y[k] corresponds to X sequence starting at starts[k]
    Y = data[window + HORIZON - 1 : window + HORIZON - 1 + len(starts), close_idx][:, np.newaxis]
    Y = 1000*Y

    # --- Build timestamp arrays if provided ---
    X_timestamps = None
    Y_timestamps = None
    if timestamps is not None:
        # For each sequence, store the timestamps of the input window
        X_timestamps = np.stack([timestamps[i:i+window] for i in starts], axis=0)
        # For Y, store the timestamp of the target value (HORIZON steps after last X)
        Y_timestamps = timestamps[window + HORIZON - 1 : window + HORIZON - 1 + len(starts)]

    return X, Y, X_timestamps, Y_timestamps

def split(array, train_frac, val_frac):
    train_size = int(len(array) * train_frac)
    val_size = int(len(array) * val_frac)

    train_set = array[:train_size]
    val_set = array[train_size:train_size + val_size]
    test_set = array[train_size + val_size:]

    return train_set, val_set, test_set

def normalize_wrt_train(train, val, test, eps=1e-8):
    """
    Normalize datasets (train, val, test) using mean/std from training set only.
    Works for arrays of shape:
      - (N, F)
      - (N, T, F)
      - (B, S, T, F)
    Normalization is applied feature-wise across *all* sequences and timesteps.

    Returns:
        train_norm, val_norm, test_norm
    """
    # Determine which axes to average over (all but last)
    axes = tuple(range(train.ndim - 1))

    # Compute feature-wise mean/std using train set only
    mu = np.mean(train, axis=axes, keepdims=True)
    sigma = np.std(train, axis=axes, keepdims=True) + eps

    # Apply normalization
    train_norm = (train - mu) / sigma
    val_norm   = (val   - mu) / sigma
    test_norm  = (test  - mu) / sigma

    return train_norm, val_norm, test_norm, mu.squeeze(), sigma.squeeze()

def verify_normalization(train_set, val_set, test_set):
    """
    Prints feature-wise means and stds for any-shaped datasets.
    Works for arrays of shape:
        (N, F), (N, T, F), or (B, S, T, F)
    """
    print("Number of features:", train_set.shape[-1])

    def summarize(name, arr):
        # Average over all axes except the last (features)
        axes = tuple(range(arr.ndim - 1))
        mean = np.mean(arr, axis=axes)
        std = np.std(arr, axis=axes)
        print(f"{name} means: {[f'{x:.3f}' for x in mean]}")
        print(f"{name} stds:  {[f'{x:.3f}' for x in std]}")

    summarize("Train", train_set)
    summarize("Val", val_set)
    summarize("Test", test_set)
    print("\n")


#### Last step, Batching ####
def create_batches(X, Y, batch_size):
    """Return list of (X_batch, Y_batch) tuples."""
    n = X.shape[0]
    X_batches = []
    Y_batches = []
    Last_X_batch = None
    Last_Y_batch = None
    for start in range(0, n, batch_size):
        end = start + batch_size
        X_batches.append(X[start:end])
        Y_batches.append(Y[start:end])

    #we must store the last batch separately, because it may be smaller than batch_size, and so numpy can't convert it to an array
    Last_X_batch = X_batches[-1] 
    Last_Y_batch = Y_batches[-1]
    X_batches.pop()
    Y_batches.pop()
    
    return np.array(X_batches), np.array(Y_batches), np.array(Last_X_batch), np.array(Last_Y_batch)

def summarize_batched_data(X_train_batches, Y_train_batches,
                           X_val_batches,   Y_val_batches,
                           X_test_batches,  Y_test_batches):
    """
    Summarizes batched datasets (train/val/test) by printing key info:
    - number of batches
    - batch size
    - sequence length
    - number of features
    - number of total examples (excluding dropped last batch)
    - relative fractions of each split
    """

    # Infer common dimensions from the first batch
    batch_size, seq_len, num_features = X_train_batches[0].shape
    num_batches_train = len(X_train_batches)
    num_batches_val   = len(X_val_batches)
    num_batches_test  = len(X_test_batches)

    # Total examples (no last batch assumption)
    n_train = num_batches_train * batch_size
    n_val   = num_batches_val   * batch_size
    n_test  = num_batches_test  * batch_size
    total   = n_train + n_val + n_test

    # Fractions
    f_train = n_train / total
    f_val   = n_val   / total
    f_test  = n_test  / total

    # ---- Pretty printing ----
    print("\n" + "#"*65)
    print(f"{'📊 DATA SUMMARY':^65}")
    print("#"*65 + "\n")

    print(f"Number of features: {num_features}")
    print(f"Sequence length per example: {seq_len}")
    print(f"Batch size: {batch_size}")
    print()

    print("Dataset Split Summary:")
    print("-" * 65)
    print(f"{'Set':<10}{'Num Batches':<15}{'Examples':<15}{'Fraction':<15}")
    print("-" * 65)
    print(f"{'Train':<10}{num_batches_train:<15}{n_train:<15}{f_train:<15.3f}")
    print(f"{'Val':<10}{num_batches_val:<15}{n_val:<15}{f_val:<15.3f}")
    print(f"{'Test':<10}{num_batches_test:<15}{n_test:<15}{f_test:<15.3f}")
    print("-" * 65)

    print("\nExample batch shapes:")
    print(f"X_train_batches[0]: {X_train_batches[0].shape}, Y_train_batches[0]: {Y_train_batches[0].shape}")
    print(f"X_val_batches[0]:   {X_val_batches[0].shape}, Y_val_batches[0]:   {Y_val_batches[0].shape}")
    print(f"X_test_batches[0]:  {X_test_batches[0].shape}, Y_test_batches[0]:  {Y_test_batches[0].shape}")

    print("\n" + "#"*65 + "\n")






if __name__ == '__main__':
    # ============================================================
    # 1) Load data from CSV
    # ============================================================
    df = load_data('BTCUSDT_30m_10years.csv')
    #df = pd.read_csv("new_features_df.csv", index_col='close_time', parse_dates=['close_time'])

    if COMBINE_DATA:
        df2 = load_data('ETHUSDT_30m_10years.csv')


    # ============================================================
    # 2) Add New Technical Indicators and compute log returns
    # ============================================================

    
    arr, col_names, close_idx, timestamps = feature_engineer1(df)
   
    NUM_FEATURES = arr.shape[1]


    if COMBINE_DATA:
        arr2, col_names2, _, timestamps2 = feature_engineer2(df2)
      

    # print("stopping here to check data shapes... can comment out sys.exit(0) to continue to training")
    # sys.exit(0)
    print("\n#####################################################\n")



    """ 
    If we shuffle blocks, we do:
    - BLOCK --> SEQUENCE BLOCKS --> SHUFFLE --> SPLIT --> NORM --> BATCH
    """
    if BLOCK_SHUFFLE:

        blocked_Xs, blocked_Ys = block_and_sequence(arr, BLOCK_SIZE, close_idx)

        if COMBINE_DATA:
            blocked_Xs2, blocked_Ys2 = block_and_sequence(arr2, BLOCK_SIZE, close_idx)
            combined_block_Xs = np.concatenate((blocked_Xs, blocked_Xs2), axis=0)
            combined_block_Ys = np.concatenate((blocked_Ys, blocked_Ys2), axis=0)
            shuffled_combined_blocked_Xs, shuffled_combined_blocked_Ys = shuffle_blocks(combined_block_Xs, combined_block_Ys)

            X = shuffled_combined_blocked_Xs
            Y = shuffled_combined_blocked_Ys

        else: 
            shuffled_blocked_Xs, shuffled_blocked_Ys = shuffle_blocks(blocked_Xs, blocked_Ys) # shuffled_blocked_Xs size (num_blocks, num_sequences, sequence_len, num_features)  
                                                                                            # shuffled_blocked_Ys size (num_blocks, num_sequences, num_features)  
            X = shuffled_blocked_Xs
            Y = shuffled_blocked_Ys

        #Each is (num_blocks, sequences_per_block, sequence_length, num_features)
        X_train, X_val, X_test = split(X, train_frac, val_frac)
        Y_train, Y_val, Y_test = split(Y, train_frac, val_frac)

        #combine all blocks - they are no longer needed
        #becomes (num_sequences, sequence_length, num_features)
        X_train = X_train.reshape(-1, X_train.shape[2], X_train.shape[3])
        Y_train = Y_train.reshape(-1, Y_train.shape[2])
        X_val   = X_val.reshape(-1, X_val.shape[2], X_val.shape[3])
        Y_val   = Y_val.reshape(-1, Y_val.shape[2])
        X_test  = X_test.reshape(-1, X_test.shape[2], X_test.shape[3])
        Y_test  = Y_test.reshape(-1, Y_test.shape[2])

        X_train, X_val, X_test, mu_x, sigma_x = normalize_wrt_train(X_train, X_val, X_test)
        Y_train, Y_val, Y_test, mu_y, sigma_y = normalize_wrt_train(Y_train, Y_val, Y_test)


        print(X_train.shape, X_val.shape, X_test.shape)

        
    # No shuffling
    else:
        # ============================================================
        # 4) Split, Normalize, and Sequence
        # ============================================================
        """
        if no shuffling, we do:
        - SPLIT --> NORM --> SEQUENCE --> BATCH
        """

        """
        If DENOISING = True,
        - SPLIT --> SEQUENCE --> DENOISE --> LOG RETURNS --> NORM --> BATCH
        - We must perform the wavelet on each sequence
        - We must perform the wavelet prior to normalization. 
        - Therefore, we must normalize after sequencing
        """
        
        train, val, test = split(arr, train_frac, val_frac)
        timestamps_train, timestamps_val, timestamps_test = split(timestamps, train_frac, val_frac)

        if COMBINE_DATA:
            train2, val2, test2 = split(arr2, train_frac, val_frac)
            timestamps_train2, timestamps_val2, timestamps_test2 = split(timestamps2, train_frac, val_frac)
            train = np.concatenate((train, train2), axis=0)
            val = np.concatenate((val, val2), axis=0)
            test = np.concatenate((test, test2), axis=0)
            timestamps_train = np.concatenate((timestamps_train, timestamps_train2), axis=0)
            timestamps_val = np.concatenate((timestamps_val, timestamps_val2), axis=0)
            timestamps_test = np.concatenate((timestamps_test, timestamps_test2), axis=0)


        ## SEQUENCE DATA
        print("Starting Sequencing...")
        X_train, Y_train, X_train_ts, Y_train_ts = sequence(train, SEQ_LEN, HORIZON, close_idx, timestamps_train)
        X_val, Y_val, X_val_ts, Y_val_ts = sequence(val, SEQ_LEN, HORIZON, close_idx, timestamps_val)
        X_test, Y_test, X_test_ts, Y_test_ts = sequence(test, SEQ_LEN, HORIZON, close_idx, timestamps_test)
        print("3 Train sequencing complete!")


        ## Normalize DATA
        X_train, X_val, X_test, mu_x, sigma_x = normalize_wrt_train(X_train, X_val, X_test)
        verify_normalization(X_train, X_val, X_test) #prints out summary

         # Save normalization statistics
        mu_sigma_df = pd.DataFrame({
            'mu': mu_x,
            'sigma': sigma_x
        })
        mu_sigma_df.to_csv('mu_sigma_df.csv', index=False)
        print("Saved normalization statistics to mu_sigma_df.csv")

    


    # ============================================================
    # 7) Create Batches of size `batch_size`
    # ============================================================
    print("starting batching...")
    X_train_batches, Y_train_batches, _, _ = create_batches(X_train, Y_train, batch_size)     #Ignore last batches
    X_val_batches, Y_val_batches, _, _ = create_batches(X_val, Y_val, batch_size)
    X_test_batches, Y_test_batches, _, _ = create_batches(X_test, Y_test, batch_size)

    # Batch timestamps (for human verification, not model input)
    X_train_ts_batches, Y_train_ts_batches, _, _ = create_batches(X_train_ts, Y_train_ts, batch_size)
    X_val_ts_batches, Y_val_ts_batches, _, _ = create_batches(X_val_ts, Y_val_ts, batch_size)
    X_test_ts_batches, Y_test_ts_batches, _, _ = create_batches(X_test_ts, Y_test_ts, batch_size)
    print("batching complete")

    summarize_batched_data(X_train_batches, Y_train_batches,
                            X_val_batches,   Y_val_batches,
                            X_test_batches,  Y_test_batches)

    



    # ============================================================
    # 8) Save Data
    # ============================================================
    fname = f"preprocessed_data{'_COMBINED' if COMBINE_DATA else ''}{'_SHUFFLED' if BLOCK_SHUFFLE else ''}.npz"

    np.savez_compressed(fname,
        X_train_batches=X_train_batches, Y_train_batches=Y_train_batches,
        X_val_batches=X_val_batches, Y_val_batches=Y_val_batches,
        X_test_batches=X_test_batches, Y_test_batches=Y_test_batches,
        X_train_ts_batches=X_train_ts_batches, Y_train_ts_batches=Y_train_ts_batches,
        X_val_ts_batches=X_val_ts_batches, Y_val_ts_batches=Y_val_ts_batches,
        X_test_ts_batches=X_test_ts_batches, Y_test_ts_batches=Y_test_ts_batches,
    )   

    print(f"Saved {fname}")




















# ============================================================
# 6) Analyze variance of signals
# ============================================================

# # === Parameters ===
# WINDOW = 1000  # smaller window (~21 days for 30-min data)
# FEATURES = ['log_return_close', 'volume', 'number_of_trades', 'BB_width']
# CSV_PATH = "BTCUSDT_30m_10years.csv"

# # === Load data ===
# df = pd.read_csv(CSV_PATH)
# df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
# df = df.sort_values('close_time')

# # === Compute Bollinger Band width (if not in dataset) ===
# if 'BB_width' not in df.columns:
#     df['BB_width'] = (df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0']) / (df['BBM_20_2.0_2.0'] + 1e-8)

# # === Compute log returns ===
# for col in ['open', 'high', 'low', 'close']:
#     df[f'log_return_{col}'] = np.log(df[col] / df[col].shift(1))
# df.fillna(0, inplace=True)

# # === Rolling standard deviation ===
# vol_df = df[[f for f in FEATURES if f in df.columns]].rolling(WINDOW).std()

# # Drop NaNs that appear at the start
# vol_df = vol_df.dropna()
# df = df.loc[vol_df.index]

# # === Train/Val/Test split markers ===
# train_frac, val_frac, test_frac = 0.9, 0.05, 0.05
# n = len(df)
# train_end = int(train_frac * n)
# val_end = int((train_frac + val_frac) * n)
# split_dates = [df.iloc[train_end]['close_time'], df.iloc[val_end]['close_time']]

# # === Plot ===
# plt.figure(figsize=(12, 6))
# for f in vol_df.columns:
#     plt.plot(df['close_time'], vol_df[f], label=f)

# for i, split_date in enumerate(split_dates):
#     plt.axvline(split_date, color='red', linestyle='--', alpha=0.7)
#     plt.text(split_date, plt.ylim()[1]*0.9, ['Train/Val', 'Val/Test'][i],
#              rotation=90, color='red', ha='right', va='top')

# plt.title(f"Rolling Volatility (Window={WINDOW} samples ≈ {WINDOW * 30 / 60 / 24:.1f} days)")
# plt.xlabel("Date")
# plt.ylabel("Rolling Std (Volatility)")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()



#TODO: 

# ADD AS LITTLE EXTRA CODE AS POSSIBLE: rely on my current code and functionality, but simply re-organize and re-parameterize it.
# - turn log returns into its own function
# - parameterize/edit denoise function 
# - Edits to 01_Retrive_API_Data.py
#     - package code functionality into individual helper functions that are called. Logically organize code into functions
#     - move fetch_google_trends() from 02_Data_Processing to the 01_Retrive_API_Data
#     - add top-level boolean parameter INCLUDE_GOOGLE_TRENDS 