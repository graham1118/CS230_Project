import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
from matplotlib import pyplot as plt
import os
import sys
from typing import Tuple
import time


# ============================================================
# Configuration
# ============================================================
results_save_path = "exp_results.pkl"
plot_save_path_suffix = "_loss_plot.jpg"
NPZ_PATH = "../Data Processing/preprocessed_data.npz"  # produced by your preprocessing script
COL_NAMES_PATH = "../Data Processing/column_names.csv"
SAVE_PATH = "cur_best_model.pth"
SEED = 0 

# ============================================================
# Hyperparameters 
# ============================================================
"""
Try:
1) Different SEQ_LENs 
2) Different HORIZONs
3) Different NUM_GRU_LAYERS
4) Different LEARNING_RATEs
"""

INIT_LEARNING_RATE = 1e-3          # smaller LR helps stability
NUM_EPOCHS = 40
NUM_GRU_LAYERS = 2
GRU_HIDDEN_SIZE = 64
CLIP_GRAD_NORM = 1.0    # gradient clipping to prevent explosion
MAX_LR = 1e-3
MODEL_TO_USE = 'GRU' #or 'LSTM'
WEIGHT_DECAY = 0
DROPOUT = 0.10 #note used
USE_SCHEDULER = False    # ReduceLROnPlateau on val loss



# ----------------------------
# Kernel, Files & Reproducibility
# ----------------------------
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("USING DEVICE:", device)

def save_object_to_file(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def read_object_from_file(filepath):
    with open(filepath, "rb") as f:
        out = pickle.load(f)
    return out

def seed_everything(seed):   #ensures randomness is fixed across runs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



# ============================================
# Load & reshape preprocessed batches from NPZ
# ============================================
def load_preprocessed_data(path=NPZ_PATH):
    """
    Returns:
        dict: Dictionary containing all arrays stored in the file.
              Keys: 'X_train_batches', 'Y_train_batches', etc.
    """

    with np.load(path) as data:
        dataset = {key: data[key] for key in data.files}
    return dataset
def flatten_features_to_close(Y):
    return  Y [..., 0].unsqueeze(-1)


data = load_preprocessed_data()
feature_names = pd.read_csv(COL_NAMES_PATH)

X_train_batches = data["X_train_batches"]
Y_train_batches = data["Y_train_batches"]

X_val_batches = data["X_val_batches"]
Y_val_batches = data["Y_val_batches"]

X_test_batches = data["X_test_batches"]
Y_test_batches = data["Y_test_batches"]

print("[INFO] Converting data to torch tensors and moving to device...")
X_train_batches = torch.from_numpy(X_train_batches).to(torch.float32).to(device)
Y_train_batches = torch.from_numpy(Y_train_batches).to(torch.float32).to(device)
X_val_batches = torch.from_numpy(X_val_batches).to(torch.float32).to(device)
Y_val_batches   = torch.from_numpy(Y_val_batches).to(torch.float32).to(device)
X_test_batches  = torch.from_numpy(X_test_batches).to(torch.float32).to(device)
Y_test_batches  = torch.from_numpy(Y_test_batches).to(torch.float32).to(device)



#set target to be only close price, rather than all features
# Y_train_batches = flatten_features_to_close(Y_train_batches)
# Y_val_batches = flatten_features_to_close(Y_val_batches)
# Y_test_batches = flatten_features_to_close(Y_test_batches)


num_train_batches = X_train_batches.shape[0]
num_val_batches = X_val_batches.shape[0]
num_test_batches = X_test_batches.shape[0]
BATCH_SIZE = X_train_batches.shape[1]  #assumes all batches have same size
SEQ_LEN = X_train_batches.shape[2]
NUM_FEATURES = X_train_batches.shape[3]
GRU_OUTPUT_SIZE = Y_train_batches.shape[-1]


print("############### META INFO ####################")
print("features: ", feature_names, '\n\n')
print("(num_batches, batch_size, seq_len, num_features):")
print("Train batches:", X_train_batches.shape)
print("Validation batches:", X_val_batches.shape)
print("Test batches:", X_test_batches.shape)
print("Training Y shape:", Y_train_batches.shape)
print(f"batch_size = {BATCH_SIZE}, seq_len = {SEQ_LEN}, num_features = {NUM_FEATURES}")
print("##############################################\n\n")

# print("######## SANITY CHECK: CONFIRM NORMS #########")
# # Flatten across batch and batch_size dimensions
# X_flat = X_train_batches.reshape(-1, X_train_batches.shape[2], X_train_batches.shape[3])  # shape: (num_batches*batch_size, seq_len, num_features)
# X_unique = X_flat.reshape(-1, X_train_batches.shape[3])  # shape: (total_timesteps, num_features)
# feature_means = np.mean(X_unique, axis=0)
# feature_stds = np.std(X_unique, axis=0)
# print("feature means", [f"{x:.3f}" for x in feature_means])
# print("feature_stds", [f"{x:.3f}" for x in feature_stds])
# print("these won't be exactly 0 or 1 because they are only for one batch, not all of them")
# print("##############################################\n\n")


# print("stopping here to check data shapes... can comment out sys.exit(0) to continue to training")
# sys.exit(0)

############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################



# ==================================
#  GRU for regression (scalar)
# ==================================

# Instantiate model. Hidden_size is the size of the hidden state vector, h_t. Seeting hidden_size > input_size allows model to learn more complex patterns, 
# and is effectively equivalent to first passing the input through a linear layer (encoder) to increase its dimensionality before feeding it to the GRU.
#At its simplest, an encoder is just a linear layer that maps the input to a higher-dimensional space.
#A more complex encoder could involve multiple layers, non-linear activations, etc.




class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=NUM_GRU_LAYERS):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # use last timestep output
        #out = self.fc(out.mean(dim=1))
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=NUM_GRU_LAYERS):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # out: (batch_size, seq_length, hidden_size)
        # (h_n, c_n) are the hidden and cell states
        out, (h_n, c_n) = self.lstm(x)

        # Take the output at the last timestep
        out = self.fc(out[:, -1, :])
        return out






# ==========================
# Evaluation Functions
# ==========================
    # compute_set_metrics() - call on full dataset after training
    # compute_set_loss()    - call on full dataset after each epoch

    # PyTorch funcions like y = model(x) and loss_func() are batch-level operations, so for 
    # set-level operations you need to write loops over batches and accumulate results.

@torch.no_grad()
def compute_set_metrics(model, X, Y, device="cpu"):
    """
    Compute MAE, RMSE, and directional accuracy for an entire dataset.
    """
    model.eval() # Set model to evaluation mode, must set back to train() at the start of each epoch

    # Convert from NumPy if needed
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y, dtype=torch.float32)

    X, Y = X.to(device), Y.to(device)

    abs_sum = 0.0
    sq_sum = 0.0
    correct = 0
    total = 0
    n_samples = 0

    # iterate across leading dimension (num_batches or N)
    for xb, yb in zip(X, Y):
        pr = model(xb)
       

        abs_sum += float((pr - yb).abs().sum().item())
        sq_sum  += float(((pr - yb) ** 2).sum().item())
        n_samples += pr.numel()

        # Directional accuracy
        pred_sign = torch.sign(pr[:,0])
        true_sign = torch.sign(yb[:,0])
        mask = (true_sign != 0)
        correct += int((pred_sign[mask] == true_sign[mask]).sum().item())
        total += int(mask.sum().item())

    mae = abs_sum / max(1, n_samples)
    rmse = (sq_sum / max(1, n_samples)) ** 0.5
    sign_acc = (correct / total) if total > 0 else float("nan")

    return mae, rmse, sign_acc

@torch.no_grad()
def compute_set_loss(model, X, Y, loss_func, device="cpu"):

    model.eval()
    total_loss = 0.0
    num_samples = 0

    X, Y = X.to(device), Y.to(device)

    # Iterate over leading dimension (batches)
    for xb, yb in zip(X, Y):
        pr = model(xb)              # Forward pass
        loss = loss_func(pr, yb)    # Compute batch loss
        total_loss += float(loss.item()) * xb.shape[0] # scale by batch size
        num_samples += xb.shape[0]

    avg_loss = total_loss / max(1, num_samples)
    return avg_loss, total_loss

@torch.no_grad()
def compute_set_loss_close(model, X, Y, loss_func, device="cpu"):
    """
    Compute total and average loss over the dataset, 
    but only for the first feature of the model's output.
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0

    X, Y = X.to(device), Y.to(device)

    # Iterate over leading dimension (batches)
    for xb, yb in zip(X, Y):
        pr = model(xb)                    # Forward pass
        pr_first = pr[..., 0].unsqueeze(-1)  # keep only first feature
        yb_first = yb[..., 0].unsqueeze(-1)
        loss = loss_func(pr_first, yb_first)  # Compute batch loss
        total_loss += float(loss.item()) * xb.shape[0]
        num_samples += xb.shape[0]

    avg_loss = total_loss / max(1, num_samples)
    return avg_loss, total_loss
# ==========================
# Training / evaluation loop
# ==========================

def train_model(
    model,
    optimizer,
    loss_func,
    X_train,
    Y_train,
    X_val,
    Y_val,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    device="cpu",
    scheduler=None
):

    # --- Move model to device ---
    model = model.to(device)

    # --- Storage for losses ---
    train_losses = []
    val_losses = []

    # --- Epoch loop ---
    for epoch in range(num_epochs):
        model.train()

        # === Train over all batches ===
        for xb, yb in zip(X_train, Y_train):
            # xb shape is (batch_size, seq_len, num_features)
            # yb shape is (batch_size, 1)

            xb, yb = xb.to(device), yb.to(device) #nothing changes if device is CPU
            optimizer.zero_grad()

            # Forward + Backprop
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()

            ### inspect for vanishing gradient##### 
            total_gn = 0.0
            for name, p in model.named_parameters():
                if p.grad is None: 
                    continue
                g = p.grad.detach()
                gn = g.norm().item()
                total_gn += gn**2
                # print very small grads
                if gn < 1e-8:
                    print(f"[tiny grad] {name}: {gn:.2e}")

            # PRINT GRADIENT NORMS FOR DEBUGGING
            # grad_norm = 0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         grad_norm += p.grad.detach().norm().item() ** 2
            # grad_norm = grad_norm ** 0.5
            # print(f"Grad norm: {grad_norm:.4e}")

            if scheduler is not None:
                scheduler.step() #not epoch-based, but val-loss based
           



            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()


        # --- Compute average loss across batches for the epoch ---
        train_loss, _ = compute_set_loss(model, X_train, Y_train, loss_func)
        val_loss, _ = compute_set_loss(model, X_val, Y_val, loss_func)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

       

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | "
        )
    
    # --- End of epoch loop ---
    mae_train, rmse_train, sign_acc_train = compute_set_metrics(model, X_train, Y_train)
    mae_val, rmse_val, sign_acc_val = compute_set_metrics(model, X_val, Y_val)
    train_metrics = (mae_train, rmse_train, sign_acc_train)
    val_metrics = (mae_val, rmse_val, sign_acc_val)


    # --- Return history and final metrics ---
    return train_losses, val_losses, train_metrics, val_metrics



# ==========================
# Main: load, build, train
# ==========================
if __name__ == "__main__":
    seed_everything(SEED)
    print("[INFO] CWD:", os.getcwd())
    print("[INFO] Looking for NPZ at:", os.path.abspath(NPZ_PATH))





    if MODEL_TO_USE == 'GRU':
        model = GRU(input_size=X_train_batches.shape[-1], hidden_size=GRU_HIDDEN_SIZE, output_size=GRU_OUTPUT_SIZE, num_layers=NUM_GRU_LAYERS)
    elif MODEL_TO_USE == 'LSTM':
        model = LSTM(input_size=X_train_batches.shape[-1], hidden_size=GRU_HIDDEN_SIZE, output_size=GRU_OUTPUT_SIZE, num_layers=NUM_GRU_LAYERS)
    else: 
        print("[ERROR] invalid model to use")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE) # Optimizes the params in model.parameters(), and already has beta_1, beta_2, eps defaults set
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=NUM_EPOCHS,
        steps_per_epoch=X_train_batches.shape[0],
        pct_start=0.3,            # e.g., 30% of cycle increasing
        anneal_strategy='cos',    # cosine decrease after peak
        div_factor=25.0,          # initial LR = max_lr/div_factor
        final_div_factor=1e4      # final LR ~ max_lr/final_div_factor
    )




    # 5) Train
    print("[INFO] Starting training...")
    start = time.time()
    train_losses, val_losses, train_metrics, val_metrics = train_model(
        model,
        optimizer,
        criterion,
        X_train_batches,
        Y_train_batches,
        X_val_batches,
        Y_val_batches,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        device=device,
        scheduler=scheduler if USE_SCHEDULER else None
    )
    end = time.time()
    print(f"[INFO] Training time: {end - start:.2f} seconds")

    mae_train, rmse_train, sign_acc_train = train_metrics
    mae_val, rmse_val, sign_acc_val = val_metrics

    test_loss, _ = compute_set_loss(model, X_test_batches, Y_test_batches, criterion)
    mae_test, rmse_test, sign_acc_test = compute_set_metrics(model, X_test_batches, Y_test_batches)




    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, SAVE_PATH)

    print(f"✅ Model saved to {SAVE_PATH}")




    # -------------------- #
    # 2. Pretty print summary
    # -------------------- #
    print("\n" + "="*60)
    print("🚀 TRAINING COMPLETE — SUMMARY OF RESULTS")
    print("="*60)
    print(f"{'Dataset':<12}{'Loss':>10}{'MAE':>12}{'RMSE':>12}{'Sign Acc':>12}")
    print("-"*60)
    print(f"{'Train':<12}{train_losses[-1]:>10.4f}{mae_train:>12.4f}{rmse_train:>12.4f}{sign_acc_train:>12.3f}")
    print(f"{'Val':<12}{val_losses[-1]:>10.4f}{mae_val:>12.4f}{rmse_val:>12.4f}{sign_acc_val:>12.3f}")
    print(f"{'Test':<12}{test_loss:>10.4f}{mae_test:>12.4f}{rmse_test:>12.4f}{sign_acc_test:>12.3f}")
    print("="*60 + "\n")

    # -------------------- #
    # 3. Plot training and validation loss
    # -------------------- #
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2, linestyle='--')
    plt.title("Training vs Validation Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save + show
    save_path = os.path.join(os.getcwd(), "loss_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"[INFO] Loss curve saved to: {save_path}")
