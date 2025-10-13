import numpy as np
import pandas as pd

### Parameters ###
window_size = 512
block_size = window_size * 4   # must be >= window_size
seed = 42
train_frac, val_frac, test_frac = 0.9, 0.05, 0.05
batch_size = 128  # must be power of 2 for efficient training

# 1) Load data from CSV
df = pd.read_csv('BTCUSDT_30m_10years.csv')
arr = df.to_numpy()




#### ---- 2) normalize features (per-column) ---- ####
mu = arr.mean(axis=0, keepdims=True)
sigma = arr.std(axis=0, keepdims=True) + 1e-8
arr_norm = (arr - mu) / sigma



#Plan: divide the time series into non-overlapping contiguous "superblocks" (each >= window_size), 
# generate all overlapping sequences inside each block, then randomly shuffle the list of blocks and
# assign whole blocks to train/val/test. This yields (1) overlapping sequences within a set (max examples) 
# and (2) no overlap across sets (no leakage), and (3) each set samples blocks from across the whole history 
# because we shuffle blocks before splitting.



#### ---- 3) Create superblocks with sequences of length `window_size` ---- ####
rng = np.random.default_rng(seed)
def blocks_to_sequences(data, window, block):
    """Return list of (X_block, Y_block) for non-overlapping blocks of length `block`."""
    T = data.shape[0]
    blocks = []
    for start in range(0, T, block):
        end = min(start + block, T)
        if end - start < window:
            continue
        # all sequence start indices within this block
        starts = np.arange(start, end - window + 1)
        Xb = np.stack([data[i:i+window] for i in starts], axis=0)
        Yb = np.stack([data[i+window-1] for i in starts], axis=0)
        blocks.append((Xb, Yb))
    return blocks

# create blocks (non-overlapping in time)
blocks = blocks_to_sequences(arr_norm, window_size, block_size)
if len(blocks) == 0:
    raise ValueError("No blocks produced; reduce block_size or window_size or use more data.")




#### ---- 4) Shuffle blocks and split into train/val/test sets ---- ####
perm = rng.permutation(len(blocks))
blocks = [blocks[i] for i in perm]

n_train = int(np.floor(train_frac * len(blocks)))
n_val = int(np.floor(val_frac * len(blocks)))
n_test = len(blocks) - n_train - n_val

train_blocks = blocks[:n_train]
val_blocks = blocks[n_train:n_train+n_val]
test_blocks = blocks[n_train+n_val:]

def concat_and_shuffle(block_list, rng=None):
    if not block_list:
        return np.empty((0, window_size, arr_norm.shape[1])), np.empty((0, arr_norm.shape[1]))
    X = np.concatenate([b[0] for b in block_list], axis=0)
    Y = np.concatenate([b[1] for b in block_list], axis=0)
    if rng is not None and X.shape[0] > 0:
        p = rng.permutation(X.shape[0])
        X, Y = X[p], Y[p]
    return X, Y

X_train, Y_train = concat_and_shuffle(train_blocks, rng)
X_val, Y_val = concat_and_shuffle(val_blocks, rng)
X_test, Y_test = concat_and_shuffle(test_blocks, rng)

print(f"blocks={len(blocks)}, train/val/test_blocks={n_train}/{n_val}/{n_test}")
print(f"examples train/val/test = {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")







#### ---- 5) Pad sets to be multiple of `window_size` ---- #### 
def pad_to_multiple(X, Y, batch_size, rng=None):
    """Pad X and Y with random examples to make number of examples a multiple of `multiple`."""
    n = X.shape[0]
    if n % batch_size == 0:
        return X, Y
    if rng is None:
        rng = np.random.default_rng()
    n_pad = batch_size - (n % batch_size)
    p = rng.integers(0, n, size=n_pad)
    X_pad, Y_pad = X[p], Y[p]
    X_padded = np.concatenate([X, X_pad], axis=0)
    Y_padded = np.concatenate([Y, Y_pad], axis=0)
    return X_padded, Y_padded

X_train, Y_train = pad_to_multiple(X_train, Y_train, batch_size, rng)
X_val, Y_val = pad_to_multiple(X_val, Y_val, batch_size, rng)
X_test, Y_test = pad_to_multiple(X_test, Y_test, batch_size, rng)






#### ---- 6) Create Batches of size `batch_size` ---- ####
def create_batches(X, Y, batch_size):
    """Return list of (X_batch, Y_batch) tuples."""
    n = X.shape[0]
    X_batches = []
    Y_batches = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        X_batches.append(X[start:end])
        Y_batches.append(Y[start:end])
    return X_batches, Y_batches

X_train_batches, Y_train_batches = create_batches(X_train, Y_train, batch_size)
X_val_batches, Y_val_batches = create_batches(X_val, Y_val, batch_size)
X_test_batches, Y_test_batches = create_batches(X_test, Y_test, batch_size)

#Print the number of batches in each set, the size of each batch, and the shape of each example
print(f"train/val/test num batches = {len(X_train_batches)}/{len(X_val_batches)}/{len(X_test_batches)}")
print(f"batch size = {batch_size}")
print(f"example shape = {X_train_batches[0].shape[1:]}")

#print the shape of each batch set
print(f"X_train_batches shape: {X_train_batches[0].shape}, Y_train_batches shape: {Y_train_batches[0].shape}")
print(f"X_val_batches shape: {X_val_batches[0].shape}, Y_val_batches shape: {Y_val_batches[0].shape}")
print(f"X_test_batches shape: {X_test_batches[0].shape}, Y_test_batches shape: {Y_test_batches[0].shape}")






#### ---- 7) Save preprocessed data to .npz file ---- ####
np.savez_compressed('preprocessed_data.npz',
    X_train_batches=X_train_batches, Y_train_batches=Y_train_batches,
    X_val_batches=X_val_batches, Y_val_batches=Y_val_batches,
    X_test_batches=X_test_batches, Y_test_batches=Y_test_batches,
)



