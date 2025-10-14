import numpy as np
import pandas as pd

### Parameters ###
predict_n_points_into_future = 10
sequence_length = 128
seed = 42
train_frac, val_frac, test_frac = 0.9, 0.05, 0.05
batch_size = 128  # must be power of 2 for efficient training
#column_names = ["open", "high", "low", "close", "volume", "RSI_14", "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0"]


# 1) Load data from CSV
df = pd.read_csv('BTCUSDT_30m_10years.csv')
print(f"Loaded {len(df)} rows from CSV")
print(f"Columns: {df.columns.tolist()}")
df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
print(f"Date range: {df['close_time'].min().strftime('%m/%d/%Y')} to {df['close_time'].max().strftime('%m/%d/%Y')}")  #print data range in MM/DD/YYYY format

column_names = df.columns.tolist()
column_names.remove('close_time')  # all columns except close_time
print(f"Using columns: {column_names}")
arr = df[column_names].to_numpy()


#### ---- 2) Normalize Technical Indicators and Volume ---- ####

mu = arr.mean(axis=0, keepdims=True)
sigma = arr.std(axis=0, keepdims=True) + 1e-8

RSI_norm = (arr[:, 5] - mu[:, 5]) / sigma[:, 5]  # Normalize only RSI column
df['RSI_14_norm'] = RSI_norm

BB_width =  df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0']/ (df['BBM_20_2.0_2.0'] + 1e-8)
df['BB_width'] = BB_width

BB_position = (df['close'] - df['BBL_20_2.0_2.0']) / (df['BBU_20_2.0_2.0'] - df['BBL_20_2.0_2.0'] + 1e-8)
df['BB_position'] = BB_position

#We will also normalize volume to have mean 0 and std 1, because 0 volumes mean that taking volume quotients won't work
volume_norm = (arr[:, 4] - mu[:, 4]) / sigma[:, 4]  # Normalize only volume column
df['volume_norm'] = volume_norm

# drop original, un-needed indicator columns and convert to numpy array
df.drop(columns=['RSI_14', 'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0', 'BBB_20_2.0_2.0', 'BBP_20_2.0_2.0'], inplace=True)
new_column_names = df.columns.tolist()
arr = df[new_column_names].to_numpy(dtype=float)


### ---- 3) Compute log returns to normalize data ---- ###
# Taking returns centers the timeseries around 1. Taking log returns centers around 0 and improves stability of training.

# print(np.min(arr[:,0]), np.min(arr[:,1]), np.min(arr[:,2]), np.min(arr[:,3]), np.min(arr[:,4]))
# print(np.max(arr[:,0]), np.max(arr[:,1]), np.max(arr[:,2]), np.max(arr[:,3]), np.max(arr[:,4]))
# print(np.mean(arr[:,0]), np.mean(arr[:,1]), np.mean(arr[:,2]), np.mean(arr[:,3]), np.mean(arr[:,4]))
log_returns = np.log(arr[1:, :4] / arr[:-1, :4])  # log returns of 'close' price
log_returns = np.vstack([np.zeros((1, 4)), log_returns])  # first return is 0
arr_norm = np.concatenate([log_returns[:, :4], arr[:, 4:]], axis=1)  # combine log returns and normalized indicators



#Plan: divide the time series into non-overlapping contiguous "superblocks" (each >= window_size), 
# generate all overlapping sequences inside each block, then randomly shuffle the list of blocks and
# assign whole blocks to train/val/test. This yields (1) overlapping sequences within a set (max examples) 
# and (2) no overlap across sets (no leakage), and (3) each set samples blocks from across the whole history 
# because we shuffle blocks before splitting.

print("\n#####################################################\n")

#### ---- 4) Split Data into Train/Val/Test ---- ####
train_size = int(len(arr) * train_frac)
val_size = int(len(arr) * val_frac)
test_size = len(arr) - train_size - val_size

train_set = arr[:train_size]
val_set = arr[train_size:train_size + val_size]
test_set = arr[train_size + val_size:]


#### ---- 5) Sequence Data with no-overlap between sets ---- ####
def sequence(data, window, predict_n_points_into_future):
    """Return list of (X, Y) for all overlapping sequences of length `window`."""
    T = data.shape[0]
    starts = np.arange(0, T - window - predict_n_points_into_future)
    X = np.stack([data[i:i+window] for i in starts], axis=0)
    Y = np.stack([data[i+window + predict_n_points_into_future] for i in starts], axis=0)
    return X, Y



X_train, Y_train = sequence(train_set, sequence_length, predict_n_points_into_future)
X_val, Y_val = sequence(train_set, sequence_length, predict_n_points_into_future)
X_test, Y_test = sequence(train_set, sequence_length, predict_n_points_into_future)


print(f"Number of examples in train/val/test = {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
print(f"Train set fraction: {X_train.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0]):.3f}")
print(f"Val set fraction: {X_val.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0]):.3f}")
print(f"Test set fraction: {X_test.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0]):.3f}")
print("Each row of X is 1 input sequence of length:", X_train.shape[1])
print("\n#####################################################\n")


#### ---- 5) Pad sets to be multiple of `window_size` ---- #### 
# def pad_to_multiple(X, Y, batch_size, rng=None):
#     """Pad X and Y with random examples to make number of examples a multiple of `multiple`."""
#     n = X.shape[0]
#     if n % batch_size == 0:
#         return X, Y
#     if rng is None:
#         rng = np.random.default_rng()
#     n_pad = batch_size - (n % batch_size)
#     p = rng.integers(0, n, size=n_pad)
#     X_pad, Y_pad = X[p], Y[p]
#     X_padded = np.concatenate([X, X_pad], axis=0)
#     Y_padded = np.concatenate([Y, Y_pad], axis=0)
#     return X_padded, Y_padded

# X_train, Y_train = pad_to_multiple(X_train, Y_train, batch_size, rng)
# X_val, Y_val = pad_to_multiple(X_val, Y_val, batch_size, rng)
# X_test, Y_test = pad_to_multiple(X_test, Y_test, batch_size, rng)






#### ---- 6) Create Batches of size `batch_size` ---- ####
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

    Last_X_batch = X_batches[-1]
    Last_Y_batch = Y_batches[-1]
    X_batches.pop()
    Y_batches.pop()
    
    return np.array(X_batches), np.array(Y_batches), np.array(Last_X_batch), np.array(Last_Y_batch)


#Keep batches as python lists of arrays, so that the last batch can be smaller than batch_size if needed
X_train_batches, Y_train_batches, Last_X_train_batch, Last_Y_train_batch = create_batches(X_train, Y_train, batch_size)
X_val_batches, Y_val_batches, Last_X_val_batch, Last_Y_val_batch = create_batches(X_val, Y_val, batch_size)
X_test_batches, Y_test_batches, Last_X_test_batch, Last_Y_test_batch = create_batches(X_test, Y_test, batch_size)

#Print the number of batches in each set, the size of each batch, and the shape of each example
print(f"train/val/test num batches = {len(X_train_batches)}/{len(X_val_batches)}/{len(X_test_batches)}")
print(f"batch size = {batch_size}")

#print the shape of each batch set
print(f"X_train_batches shape: {X_train_batches[0].shape}, Y_train_batches shape: {Y_train_batches[0].shape}")
print(f"X_val_batches shape: {X_val_batches[0].shape}, Y_val_batches shape: {Y_val_batches[0].shape}")
print(f"X_test_batches shape: {X_test_batches[0].shape}, Y_test_batches shape: {Y_test_batches[0].shape}")
print(f"Size of last batch in train/val/test = {Last_X_train_batch.shape[0]}/{Last_X_val_batch.shape[0]}/{Last_Y_test_batch.shape[0]}")
print(f"Size of last batch")
print("\n#####################################################\n")





#### ---- 7) Save preprocessed data to .npz file ---- ####
np.savez_compressed('preprocessed_data.npz',
    X_train_batches=X_train_batches, Y_train_batches=Y_train_batches,
    Last_X_train_batch=Last_X_train_batch, Last_Y_train_batch=Last_Y_train_batch,
    X_val_batches=X_val_batches, Y_val_batches=Y_val_batches,
    Last_X_val_batch=Last_X_val_batch, Last_Y_val_batch=Last_Y_val_batch,
    X_test_batches=X_test_batches, Y_test_batches=Y_test_batches,
    Last_X_test_batch=Last_X_test_batch, Last_Y_test_batch=Last_Y_test_batch,
)