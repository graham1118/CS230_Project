# Training file for Attention-based Sequence-to-Sequence forecasting

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

# Import model architecture
from model import create_model, AttentionSeq2Seq

# ============================================================
# Configuration
# ============================================================
SEED = 42

# File paths
COMBINE_DATA = False
BLOCK_SHUFFLE = False
NPZ_FNAME = f"preprocessed_data{'_COMBINED' if COMBINE_DATA else ''}{'_SHUFFLED' if BLOCK_SHUFFLE else ''}.npz"
NPZ_PATH = NPZ_FNAME  # Assuming we're in the same directory after running data_processing.py
COL_NAMES_PATH = "column_names.csv"
MU_SIGMA_PATH = "mu_sigma_df.csv"
SAVE_PATH = "attention_model_best.pth"

# Model structure
MODEL_TYPE = 'LSTM'  # 'LSTM' or 'GRU'
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LATENT_SIZE = 16
USE_ATTENTION = True
ATTENTION_TEMPERATURE = 1.0
ENCODER_LENGTH_P = 32  # Must match data_processing.py
DECODER_LENGTH_T = 16   # Must match data_processing.py
T1 = 6                  # Validation prefix
T2 = 10                 # Final forecast

# Training hyperparameters
INIT_LEARNING_RATE = 2e-4
MAX_LR = 2e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 60
CLIP_GRAD_NORM = 1.0
DROPOUT = 0.2
USE_SCHEDULER = True
TEMPERATURE = 1.0  # Sampling temperature during inference
BATCH_SIZE = 128   # Must match data_processing.py

# Scheduled sampling
SCHEDULED_SAMPLING_START_EPOCH = 5
SCHEDULED_SAMPLING_END_PROB = 0.5
MIN_STD_CLAMP = 1e-3

# Checkpointing
CHECKPOINT_EVERY = 10

# Device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")


# ============================================================
# Reproducibility
# ============================================================
def seed_everything(seed):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# File I/O
# ============================================================
def save_object_to_file(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def read_object_from_file(filepath):
    with open(filepath, "rb") as f:
        out = pickle.load(f)
    return out


# ============================================================
# Data Loading
# ============================================================
def load_preprocessed_data(path=NPZ_PATH):
    """
    Load preprocessed data from NPZ file.

    Returns
    -------
    dict : Dictionary containing all arrays stored in the file.
           Keys: 'X_train_batches', 'Y_train_batches', etc.
    """
    with np.load(path) as data:
        dataset = {key: data[key] for key in data.files}
    return dataset


# ============================================================
# Loss Function: Gaussian Negative Log-Likelihood
# ============================================================
def gaussian_nll_loss(mean, std, target):
    """
    Compute Gaussian Negative Log-Likelihood loss.

    Loss = 0.5 * sum_d[(y - mu)^2 / sigma^2 + log(sigma^2) + log(2*pi)]

    Parameters
    ----------
    mean : torch.Tensor
        Predicted mean, shape [batch_size, T, D] or [batch_size, D]
    std : torch.Tensor
        Predicted std, shape [batch_size, T, D] or [batch_size, D]
    target : torch.Tensor
        Ground truth, shape [batch_size, T, D] or [batch_size, D]

    Returns
    -------
    loss : torch.Tensor
        Scalar loss (mean over batch and timesteps, sum over features)
    """
    # Prevent division by zero and log of zero
    std = std.clamp(min=MIN_STD_CLAMP)

    # Compute NLL components
    # (y - mu)^2 / sigma^2
    squared_error = ((target - mean) ** 2) / (std ** 2)

    # log(sigma^2) = 2 * log(sigma)
    log_variance = 2 * torch.log(std)

    # Constant term: log(2*pi)
    log_2pi = np.log(2 * np.pi)

    # Total NLL per feature
    nll = 0.5 * (squared_error + log_variance + log_2pi)

    # Sum over features, mean over batch and timesteps
    loss = nll.sum(dim=-1).mean()

    return loss


# ============================================================
# Evaluation Metrics
# ============================================================
@torch.no_grad()
def compute_directional_accuracy(predictions, targets, close_idx):
    """
    Compute directional accuracy (sign agreement) for close price predictions.

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted values, shape [batch_size, T, D]
    targets : torch.Tensor
        True values, shape [batch_size, T, D]
    close_idx : int
        Index of close price feature

    Returns
    -------
    accuracy : float
        Directional accuracy (0-1)
    """
    # Extract close prices
    pred_close = predictions[:, :, close_idx]  # [batch_size, T]
    true_close = targets[:, :, close_idx]      # [batch_size, T]

    # Compute signs
    pred_sign = torch.sign(pred_close)
    true_sign = torch.sign(true_close)

    # Only consider non-zero true values
    mask = (true_sign != 0)

    if mask.sum() == 0:
        return 0.0

    # Count correct signs
    correct = (pred_sign[mask] == true_sign[mask]).sum().item()
    total = mask.sum().item()

    accuracy = correct / total

    return accuracy


@torch.no_grad()
def compute_r2_score(predictions, targets, close_idx):
    """
    Compute R^2 score for close price predictions.

    R^2 = 1 - (SS_res / SS_tot)

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted values, shape [batch_size, T, D]
    targets : torch.Tensor
        True values, shape [batch_size, T, D]
    close_idx : int
        Index of close price feature

    Returns
    -------
    r2 : float
        R^2 score
    """
    # Extract close prices
    pred_close = predictions[:, :, close_idx].flatten()
    true_close = targets[:, :, close_idx].flatten()

    # Compute residual sum of squares
    ss_res = ((true_close - pred_close) ** 2).sum()

    # Compute total sum of squares
    ss_tot = ((true_close - true_close.mean()) ** 2).sum()

    # R^2
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return r2.item()


@torch.no_grad()
def evaluate_model(model, X_batches, Y_batches, close_idx, device):
    """
    Evaluate model on a dataset.

    Parameters
    ----------
    model : AttentionSeq2Seq
        The model to evaluate
    X_batches : torch.Tensor
        Input batches, shape [num_batches, batch_size, P, D]
    Y_batches : torch.Tensor
        Target batches, shape [num_batches, batch_size, T, D]
    close_idx : int
        Index of close price feature
    device : str
        Device to use

    Returns
    -------
    avg_loss : float
        Average NLL loss
    dir_acc : float
        Directional accuracy
    r2 : float
        R^2 score
    avg_std : float
        Average predicted std (for monitoring collapse)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_stds = []
    num_samples = 0

    for xb, yb in zip(X_batches, Y_batches):
        xb, yb = xb.to(device), yb.to(device)
        batch_size = xb.size(0)

        # Forward pass (no teacher forcing during evaluation)
        means, stds, samples, _ = model(xb, T=DECODER_LENGTH_T, temperature=1.0,
                                        ground_truth=None, teacher_forcing_prob=0.0)

        # Compute loss
        loss = gaussian_nll_loss(means, stds, yb)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        # Store for metrics
        all_predictions.append(means.cpu())
        all_targets.append(yb.cpu())
        all_stds.append(stds.cpu())

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_stds = torch.cat(all_stds, dim=0)

    # Compute metrics
    avg_loss = total_loss / num_samples
    dir_acc = compute_directional_accuracy(all_predictions, all_targets, close_idx)
    r2 = compute_r2_score(all_predictions, all_targets, close_idx)
    avg_std = all_stds.mean().item()

    return avg_loss, dir_acc, r2, avg_std


# ============================================================
# Training Loop
# ============================================================
def train_model(model, optimizer, scheduler, X_train, Y_train, X_val, Y_val,
                num_epochs, close_idx, device):
    """
    Train the model with scheduled sampling and NLL loss.

    Parameters
    ----------
    model : AttentionSeq2Seq
        Model to train
    optimizer : torch.optim.Optimizer
        Optimizer
    scheduler : torch.optim.lr_scheduler or None
        Learning rate scheduler
    X_train : torch.Tensor
        Training inputs, shape [num_batches, batch_size, P, D]
    Y_train : torch.Tensor
        Training targets, shape [num_batches, batch_size, T, D]
    X_val : torch.Tensor
        Validation inputs
    Y_val : torch.Tensor
        Validation targets
    num_epochs : int
        Number of epochs
    close_idx : int
        Index of close price
    device : str
        Device

    Returns
    -------
    history : dict
        Training history with losses and metrics
    """
    model = model.to(device)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dir_acc': [],
        'val_r2': [],
        'avg_std': [],
        'learning_rate': [],
        'teacher_forcing_prob': []
    }

    best_val_loss = float('inf')
    best_state = None

    print("="*70)
    print(f"{'STARTING TRAINING':^70}")
    print("="*70)
    print(f"Num epochs: {num_epochs}")
    print(f"Num train batches: {len(X_train)}")
    print(f"Num val batches: {len(X_val)}")
    print(f"Batch size: {X_train.shape[1]}")
    print(f"Encoder length (P): {ENCODER_LENGTH_P}")
    print(f"Decoder length (T): {DECODER_LENGTH_T}")
    print("="*70 + "\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()

        # Compute scheduled sampling probability (linear schedule)
        if epoch < SCHEDULED_SAMPLING_START_EPOCH:
            teacher_forcing_prob = 1.0
        else:
            progress = (epoch - SCHEDULED_SAMPLING_START_EPOCH) / (num_epochs - SCHEDULED_SAMPLING_START_EPOCH)
            teacher_forcing_prob = max(1.0 - progress * (1.0 - SCHEDULED_SAMPLING_END_PROB), SCHEDULED_SAMPLING_END_PROB)

        # Shuffle training batches
        num_batches = X_train.shape[0]
        perm = torch.randperm(num_batches)
        X_train_shuffled = X_train[perm]
        Y_train_shuffled = Y_train[perm]

        # Training loop with progress bar
        train_loss_epoch = 0.0
        train_samples = 0

        pbar = tqdm(zip(X_train_shuffled, Y_train_shuffled), total=len(X_train_shuffled),
                   desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_idx, (xb, yb) in enumerate(pbar):
            xb, yb = xb.to(device), yb.to(device)
            batch_size = xb.size(0)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with scheduled sampling
            means, stds, samples, _ = model(xb, T=DECODER_LENGTH_T, temperature=1.0,
                                           ground_truth=yb, teacher_forcing_prob=teacher_forcing_prob)

            # Compute loss
            loss = gaussian_nll_loss(means, stds, yb)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)

            # Optimizer step
            optimizer.step()

            # Update progress bar
            train_loss_epoch += loss.item() * batch_size
            train_samples += batch_size
            pbar.set_postfix({'loss': loss.item()})

        # Compute epoch metrics
        train_loss_avg = train_loss_epoch / train_samples

        # Validation
        val_loss, val_dir_acc, val_r2, val_avg_std = evaluate_model(
            model, X_val, Y_val, close_idx, device
        )

        # Update scheduler
        if USE_SCHEDULER and scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Store history
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss)
        history['val_dir_acc'].append(val_dir_acc)
        history['val_r2'].append(val_r2)
        history['avg_std'].append(val_avg_std)
        history['learning_rate'].append(current_lr)
        history['teacher_forcing_prob'].append(teacher_forcing_prob)

        # Epoch time
        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dir Acc: {val_dir_acc:.3f} | "
              f"Val R²: {val_r2:.3f} | "
              f"Avg Std: {val_avg_std:.4f} | "
              f"TF Prob: {teacher_forcing_prob:.2f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ’ New best validation loss: {best_val_loss:.4f}")

        # Periodic checkpointing
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            checkpoint_path = SAVE_PATH.replace('.pth', f'_epoch{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, history, checkpoint_path)
            print(f"  ’ Checkpoint saved: {checkpoint_path}")

        print()

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f" Loaded best model with validation loss: {best_val_loss:.4f}\n")

    return history


# ============================================================
# Checkpointing
# ============================================================
def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, history, path):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'best_val_loss': val_loss,
        'config': {
            'MODEL_TYPE': MODEL_TYPE,
            'HIDDEN_SIZE': HIDDEN_SIZE,
            'NUM_LAYERS': NUM_LAYERS,
            'LATENT_SIZE': LATENT_SIZE,
            'ENCODER_LENGTH_P': ENCODER_LENGTH_P,
            'DECODER_LENGTH_T': DECODER_LENGTH_T,
            'T1': T1,
            'T2': T2,
            'DROPOUT': DROPOUT,
            'ATTENTION_TEMPERATURE': ATTENTION_TEMPERATURE,
            'MIN_STD_CLAMP': MIN_STD_CLAMP
        },
        'training_history': history
    }
    torch.save(checkpoint, path)


# ============================================================
# Visualization
# ============================================================
def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curve
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('NLL Loss', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # Directional Accuracy
    axes[0, 1].plot(history['val_dir_acc'], label='Val Dir Acc', linewidth=2, color='green')
    axes[0, 1].set_title('Validation Directional Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # R^2 Score
    axes[1, 0].plot(history['val_r2'], label='Val R²', linewidth=2, color='orange')
    axes[1, 0].set_title('Validation R² Score', fontsize=14)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('R²', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # Average Std (for mode collapse detection)
    axes[1, 1].plot(history['avg_std'], label='Avg Predicted Std', linewidth=2, color='red')
    axes[1, 1].set_title('Average Predicted Std (Mode Collapse Monitor)', fontsize=14)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Std', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Training curves saved to: {save_path}")
    plt.show()


# ============================================================
# Main Training Script
# ============================================================
if __name__ == "__main__":
    seed_everything(SEED)
    print(f"Working directory: {os.getcwd()}\n")

    # ============================================================
    # Load Data
    # ============================================================
    print("Loading preprocessed data...")
    data = load_preprocessed_data(NPZ_PATH)

    X_train_batches = torch.from_numpy(data['X_train_batches']).float()
    Y_train_batches = torch.from_numpy(data['Y_train_batches']).float()
    X_val_batches = torch.from_numpy(data['X_val_batches']).float()
    Y_val_batches = torch.from_numpy(data['Y_val_batches']).float()
    X_test_batches = torch.from_numpy(data['X_test_batches']).float()
    Y_test_batches = torch.from_numpy(data['Y_test_batches']).float()

    print(f" Data loaded successfully")
    print(f"  X_train: {X_train_batches.shape}")
    print(f"  Y_train: {Y_train_batches.shape}")
    print(f"  X_val: {X_val_batches.shape}")
    print(f"  Y_val: {Y_val_batches.shape}")
    print(f"  X_test: {X_test_batches.shape}")
    print(f"  Y_test: {Y_test_batches.shape}\n")

    # Get dimensions
    NUM_FEATURES = X_train_batches.shape[-1]

    # Load column names
    col_names = pd.read_csv(COL_NAMES_PATH, header=None).squeeze().tolist()
    close_idx = col_names.index("log_return_close")
    print(f"Close price index: {close_idx}")
    print(f"Feature names: {col_names}\n")

    # ============================================================
    # Create Model
    # ============================================================
    print("Creating model...")
    model = create_model(
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        attention_temperature=ATTENTION_TEMPERATURE,
        min_std_clamp=MIN_STD_CLAMP,
        model_type=MODEL_TYPE
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Model created: {model.__class__.__name__}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")

    # ============================================================
    # Setup Optimizer and Scheduler
    # ============================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=INIT_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        scheduler = None

    # ============================================================
    # Train Model
    # ============================================================
    print("Starting training...\n")
    start_time = time.time()

    history = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        X_train=X_train_batches,
        Y_train=Y_train_batches,
        X_val=X_val_batches,
        Y_val=Y_val_batches,
        num_epochs=NUM_EPOCHS,
        close_idx=close_idx,
        device=device
    )

    total_time = time.time() - start_time
    print(f" Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")

    # ============================================================
    # Final Evaluation on All Sets
    # ============================================================
    print("="*70)
    print(f"{'FINAL EVALUATION':^70}")
    print("="*70)

    # Move test data to device
    X_test_batches = X_test_batches
    Y_test_batches = Y_test_batches

    # Evaluate on all sets
    train_loss, train_dir_acc, train_r2, _ = evaluate_model(
        model, X_train_batches, Y_train_batches, close_idx, device
    )
    val_loss, val_dir_acc, val_r2, _ = evaluate_model(
        model, X_val_batches, Y_val_batches, close_idx, device
    )
    test_loss, test_dir_acc, test_r2, _ = evaluate_model(
        model, X_test_batches, Y_test_batches, close_idx, device
    )

    # Print results
    print(f"\n{'Dataset':<12}{'Loss':>10}{'Dir Acc':>12}{'R²':>12}")
    print("-"*70)
    print(f"{'Train':<12}{train_loss:>10.4f}{train_dir_acc:>12.3f}{train_r2:>12.3f}")
    print(f"{'Val':<12}{val_loss:>10.4f}{val_dir_acc:>12.3f}{val_r2:>12.3f}")
    print(f"{'Test':<12}{test_loss:>10.4f}{test_dir_acc:>12.3f}{test_r2:>12.3f}")
    print("="*70 + "\n")

    # ============================================================
    # Save Final Model
    # ============================================================
    save_checkpoint(model, optimizer, scheduler, NUM_EPOCHS, val_loss, history, SAVE_PATH)
    print(f" Final model saved to: {SAVE_PATH}\n")

    # Save final checkpoint
    final_path = SAVE_PATH.replace('.pth', '_final.pth')
    save_checkpoint(model, optimizer, scheduler, NUM_EPOCHS, val_loss, history, final_path)
    print(f" Final checkpoint saved to: {final_path}\n")

    # ============================================================
    # Plot Training Curves
    # ============================================================
    plot_training_curves(history)

    print("\n=€ Training complete! Ready for evaluation.")
