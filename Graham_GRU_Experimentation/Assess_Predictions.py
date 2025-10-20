import torch
from torch import nn
import pandas as pd
from GRU import GRU, LSTM, GRU_HIDDEN_SIZE, GRU_OUTPUT_SIZE, NUM_GRU_LAYERS, INIT_LEARNING_RATE, NUM_FEATURES
from GRU import NPZ_PATH, COL_NAMES_PATH, SAVE_PATH
from GRU import load_preprocessed_data


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("USING DEVICE:", device)


# ============================================================
# Load in Test Data
# ============================================================
data = load_preprocessed_data(path=NPZ_PATH)
feature_names = pd.read_csv(COL_NAMES_PATH)

X_test_batches = data["X_test_batches"]
Y_test_batches = data["Y_test_batches"]

X_test_batches_torch  = torch.from_numpy(X_test_batches).to(torch.float32).to(device)
Y_test_batches_torch  = torch.from_numpy(Y_test_batches).to(torch.float32).to(device)

num_test_batches = X_test_batches.shape[0]
BATCH_SIZE = X_test_batches.shape[1]  #assumes all batches have same size
SEQ_LEN = X_test_batches.shape[2]
NUM_FEATURES = X_test_batches.shape[3]




# ============================================================
# Load saved model
# ============================================================
# Must define model architecture exactly the same way as in GRU.py
model = GRU(input_size=NUM_FEATURES, hidden_size=GRU_HIDDEN_SIZE, output_size=GRU_OUTPUT_SIZE, num_layers=NUM_GRU_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

# Load saved state
checkpoint = torch.load(SAVE_PATH, map_location='cpu')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
if checkpoint['scheduler_state_dict'] is not None:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

epoch = checkpoint['epoch']
train_loss = checkpoint['train_loss']
val_loss = checkpoint['val_loss']
print(f"Loaded model from epoch {epoch}, val_loss={val_loss:.4f}")





model.eval()
predictions = []

with torch.no_grad():
    for xb in X_test_batches_torch:
        xb = xb.to(device).float()
        pr = model(xb)
        predictions.append(pr.cpu())

# Concatenate all predictions
predictions = torch.cat(predictions, dim=0).numpy()

print("✅ Predictions shape:", predictions.shape)


mu_sigma_df = pd.read_csv("../Data Processing/mu_sigma_df.csv")
mu = mu_sigma_df["mu"]
sigma = mu_sigma_df["sigma"]
print(mu)
print(sigma)



# X_close = X_test_batches_torch[..., 0]          # take only the first feature (close)
# X_close_flat = X_close.reshape(-1, X_close.shape[-1])  # flatten across batches
# test_sequences = X_close_flat # (n, 11) matrix, most recent at bottom

# Y_close = Y_test_batches_torch[..., 0]          # take only the first feature (close)
# Y_close_flat = Y_close.reshape(-1, Y_close.shape[-1])  # flatten across batches
# actual_returns = Y_close_flat #(n, 1) matrix

print(test_sequences.shape)
print(actual_returns.shape)


###reconstruct prices 
#must pass in mean and sigma from earlier
# pred_prices = np.zeros_like(true_prices)
# pred_prices[0] = true_prices[0]  # initialize same starting price
# for t in range(1, len(true_prices)):
#     pred_prices[t] = true_prices[t-1] * np.exp(pred_logrets[t])

# return pred_prices, true_prices


for i in reversed(range(len(test_sequences))):
    print("X[-5:] - ", X_test_batches[i])




