# Evaluation file for Attention-based Sequence-to-Sequence forecasting
# Implements trajectory generation, prefix validation, and k-best selection

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Import model
from model import create_model, AttentionSeq2Seq

# Try to import DTW - if not available, use correlation fallback
try:
    from scipy.spatial.distance import euclidean
    from fastdtw import fastdtw
    DTW_AVAILABLE = True
except ImportError:
    print("Warning: fastdtw not available. Install with: pip install fastdtw")
    print("Falling back to correlation-based selection.\n")
    DTW_AVAILABLE = False


# ============================================================
# Configuration (must match training.py)
# ============================================================
# File paths
COMBINE_DATA = False
BLOCK_SHUFFLE = False
NPZ_FNAME = f"preprocessed_data{'_COMBINED' if COMBINE_DATA else ''}{'_SHUFFLED' if BLOCK_SHUFFLE else ''}.npz"
NPZ_PATH = NPZ_FNAME
COL_NAMES_PATH = "column_names.csv"
MU_SIGMA_PATH = "mu_sigma_df.csv"
LOAD_MODEL_PATH = "attention_model_best.pth"  # Can be overridden

# Model structure (will be loaded from checkpoint)
MODEL_TYPE = 'LSTM'
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LATENT_SIZE = 16
ENCODER_LENGTH_P = 32
DECODER_LENGTH_T = 16
T1 = 6   # Validation prefix
T2 = 10  # Final forecast
DROPOUT = 0.2
ATTENTION_TEMPERATURE = 1.0
MIN_STD_CLAMP = 1e-3

# Evaluation hyperparameters
NUM_SAMPLES = 50          # Number of trajectories to generate per inference
EVAL_EVERY_S = 20         # Datapoint interval for generating samples
K_BEST = 10               # Number of best samples to select
SELECTION_METRIC = 'DTW' if DTW_AVAILABLE else 'correlation'  # 'DTW' or 'correlation'
TEMPERATURE = 1.0         # Sampling temperature

# Device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")


# ============================================================
# Data Loading
# ============================================================
def load_preprocessed_data(path=NPZ_PATH):
    """Load preprocessed data from NPZ file."""
    with np.load(path) as data:
        dataset = {key: data[key] for key in data.files}
    return dataset


# ============================================================
# Model Loading
# ============================================================
def load_model(checkpoint_path, input_size, device):
    """
    Load trained model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file
    input_size : int
        Number of input features
    device : str
        Device to load model on

    Returns
    -------
    model : AttentionSeq2Seq
        Loaded model in eval mode
    config : dict
        Model configuration from checkpoint
    """
    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract config
    config = checkpoint.get('config', {})

    # Verify critical parameters
    print("\nCheckpoint configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create model with loaded config
    model = create_model(
        input_size=input_size,
        hidden_size=config.get('HIDDEN_SIZE', HIDDEN_SIZE),
        latent_size=config.get('LATENT_SIZE', LATENT_SIZE),
        num_layers=config.get('NUM_LAYERS', NUM_LAYERS),
        dropout=config.get('DROPOUT', DROPOUT),
        attention_temperature=config.get('ATTENTION_TEMPERATURE', ATTENTION_TEMPERATURE),
        min_std_clamp=config.get('MIN_STD_CLAMP', MIN_STD_CLAMP),
        model_type=config.get('MODEL_TYPE', MODEL_TYPE)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to eval mode
    model.eval()
    model.to(device)

    print(f" Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"   Validation loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}\n")

    return model, config


# ============================================================
# Trajectory Generation
# ============================================================
@torch.no_grad()
def generate_trajectories(model, x, num_samples, T, temperature, device):
    """
    Generate multiple independent trajectories from a single input.

    Parameters
    ----------
    model : AttentionSeq2Seq
        Trained model
    x : torch.Tensor
        Input sequence, shape [batch_size, P, D]
    num_samples : int
        Number of trajectories to generate
    T : int
        Number of timesteps to predict
    temperature : float
        Sampling temperature
    device : str
        Device

    Returns
    -------
    trajectories : torch.Tensor
        Generated trajectories, shape [num_samples, batch_size, T, D]
    """
    model.eval()
    x = x.to(device)
    batch_size = x.size(0)

    all_trajectories = []

    for _ in range(num_samples):
        # Generate one trajectory
        means, stds, samples, _ = model(
            x, T=T, temperature=temperature,
            ground_truth=None, teacher_forcing_prob=0.0
        )
        all_trajectories.append(samples)

    # Stack trajectories
    trajectories = torch.stack(all_trajectories, dim=0)  # [num_samples, batch_size, T, D]

    return trajectories


# ============================================================
# Shape-based Selection Metrics
# ============================================================
def compute_dtw_distance(pred_series, true_series):
    """
    Compute Dynamic Time Warping distance between two series.

    Parameters
    ----------
    pred_series : np.ndarray
        Predicted series, shape [T1]
    true_series : np.ndarray
        True series, shape [T1]

    Returns
    -------
    distance : float
        DTW distance
    """
    if not DTW_AVAILABLE:
        # Fallback to MSE
        return np.mean((pred_series - true_series) ** 2)

    distance, _ = fastdtw(pred_series, true_series, dist=euclidean)
    return distance


def compute_correlation(pred_series, true_series):
    """
    Compute Pearson correlation between two series.

    Parameters
    ----------
    pred_series : np.ndarray
        Predicted series, shape [T1]
    true_series : np.ndarray
        True series, shape [T1]

    Returns
    -------
    correlation : float
        Pearson correlation coefficient
    """
    # Handle constant series
    if np.std(pred_series) < 1e-8 or np.std(true_series) < 1e-8:
        return 0.0

    corr = np.corrcoef(pred_series, true_series)[0, 1]

    # Handle NaN
    if np.isnan(corr):
        return 0.0

    return corr


# ============================================================
# K-Best Selection and Weighted Combination
# ============================================================
def select_k_best_and_combine(trajectories, ground_truth, k_best, close_idx,
                              T1, T2, metric='DTW'):
    """
    Select k-best trajectories based on T1 prefix validation and combine for T2 prediction.

    Parameters
    ----------
    trajectories : torch.Tensor or np.ndarray
        Generated trajectories, shape [num_samples, T, D]
    ground_truth : torch.Tensor or np.ndarray
        Ground truth, shape [T, D]
    k_best : int
        Number of best samples to select
    close_idx : int
        Index of close price feature
    T1 : int
        Validation prefix length
    T2 : int
        Final forecast length
    metric : str
        'DTW' or 'correlation'

    Returns
    -------
    final_prediction : np.ndarray
        Combined prediction for full T period, shape [T, D]
    selected_indices : np.ndarray
        Indices of selected samples
    weights : np.ndarray
        Weights for selected samples
    scores : np.ndarray
        Scores for all samples
    """
    # Convert to numpy if needed
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()

    num_samples = trajectories.shape[0]
    T_total = trajectories.shape[1]

    # Extract close prices for T1 period
    pred_close = trajectories[:, :T1, close_idx]  # [num_samples, T1]
    true_close = ground_truth[:T1, close_idx]     # [T1]

    # Compute similarity scores
    scores = np.zeros(num_samples)

    for i in range(num_samples):
        if metric == 'DTW':
            # Lower is better
            scores[i] = compute_dtw_distance(pred_close[i], true_close)
        elif metric == 'correlation':
            # Higher is better, so negate
            scores[i] = -compute_correlation(pred_close[i], true_close)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Select k-best (lowest scores)
    selected_indices = np.argsort(scores)[:k_best]

    # Compute inverse-error weights
    selected_scores = scores[selected_indices]

    # For DTW: distance (lower is better) ’ inverse weight
    # For correlation: -corr (lower is better) ’ inverse weight
    epsilon = 1e-8
    raw_weights = 1.0 / (selected_scores + epsilon)

    # Normalize weights
    weights = raw_weights / raw_weights.sum()

    # Combine trajectories using weighted average
    # Apply weights across all timesteps and features
    selected_trajectories = trajectories[selected_indices]  # [k_best, T, D]

    # Weighted combination
    final_prediction = np.zeros_like(ground_truth)  # [T, D]

    for t in range(T_total):
        for d in range(ground_truth.shape[1]):
            final_prediction[t, d] = np.sum(weights * selected_trajectories[:, t, d])

    return final_prediction, selected_indices, weights, scores


# ============================================================
# Full Evaluation Pipeline
# ============================================================
def evaluate_with_prefix_validation(model, X_test, Y_test, close_idx, num_samples,
                                   k_best, eval_every_s, T1, T2, metric, temperature, device):
    """
    Evaluate model with prefix validation and k-best selection.

    Parameters
    ----------
    model : AttentionSeq2Seq
        Trained model
    X_test : torch.Tensor
        Test inputs, shape [num_batches, batch_size, P, D]
    Y_test : torch.Tensor
        Test targets, shape [num_batches, batch_size, T, D]
    close_idx : int
        Index of close price
    num_samples : int
        Number of trajectories to generate
    k_best : int
        Number of best samples to select
    eval_every_s : int
        Evaluation interval
    T1 : int
        Validation prefix length
    T2 : int
        Final forecast length
    metric : str
        Selection metric ('DTW' or 'correlation')
    temperature : float
        Sampling temperature
    device : str
        Device

    Returns
    -------
    results : dict
        Evaluation results
    """
    model.eval()

    # Flatten batches
    X_test_flat = X_test.reshape(-1, X_test.shape[2], X_test.shape[3])  # [N, P, D]
    Y_test_flat = Y_test.reshape(-1, Y_test.shape[2], Y_test.shape[3])  # [N, T, D]

    num_examples = X_test_flat.shape[0]
    T_total = DECODER_LENGTH_T

    # Storage for results
    all_predictions = []
    all_ground_truth = []
    all_selected_indices = []
    all_weights = []

    # Sample every EVAL_EVERY_S datapoints
    eval_indices = list(range(0, num_examples, eval_every_s))

    print(f"Evaluating on {len(eval_indices)} examples (every {eval_every_s} datapoints)...")

    for idx in tqdm(eval_indices, desc="Generating and selecting trajectories"):
        # Get single example
        x = X_test_flat[idx:idx+1]  # [1, P, D]
        y_true = Y_test_flat[idx]   # [T, D]

        # Generate NUM_SAMPLES trajectories
        trajectories = generate_trajectories(
            model, x, num_samples, T_total, temperature, device
        )  # [num_samples, 1, T, D]

        # Remove batch dimension
        trajectories = trajectories.squeeze(1)  # [num_samples, T, D]

        # Select k-best and combine
        final_pred, selected_idx, weights, scores = select_k_best_and_combine(
            trajectories, y_true, k_best, close_idx, T1, T2, metric
        )

        # Store results
        all_predictions.append(final_pred)
        all_ground_truth.append(y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true)
        all_selected_indices.append(selected_idx)
        all_weights.append(weights)

    # Convert to arrays
    all_predictions = np.array(all_predictions)      # [num_eval, T, D]
    all_ground_truth = np.array(all_ground_truth)    # [num_eval, T, D]

    # Compute metrics on T2 period
    predictions_T2 = all_predictions[:, T1:, close_idx]  # [num_eval, T2]
    ground_truth_T2 = all_ground_truth[:, T1:, close_idx]  # [num_eval, T2]

    # MSE on T2
    mse_T2 = np.mean((predictions_T2 - ground_truth_T2) ** 2)
    rmse_T2 = np.sqrt(mse_T2)

    # MAE on T2
    mae_T2 = np.mean(np.abs(predictions_T2 - ground_truth_T2))

    # Directional accuracy on T2
    pred_sign = np.sign(predictions_T2)
    true_sign = np.sign(ground_truth_T2)
    dir_acc_T2 = np.mean(pred_sign == true_sign)

    # R^2 on T2
    ss_res = np.sum((ground_truth_T2 - predictions_T2) ** 2)
    ss_tot = np.sum((ground_truth_T2 - ground_truth_T2.mean()) ** 2)
    r2_T2 = 1 - (ss_res / (ss_tot + 1e-8))

    # Also compute full trajectory metrics
    predictions_full = all_predictions[:, :, close_idx]
    ground_truth_full = all_ground_truth[:, :, close_idx]

    mse_full = np.mean((predictions_full - ground_truth_full) ** 2)
    mae_full = np.mean(np.abs(predictions_full - ground_truth_full))
    dir_acc_full = np.mean(np.sign(predictions_full) == np.sign(ground_truth_full))

    results = {
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'selected_indices': all_selected_indices,
        'weights': all_weights,
        'mse_T2': mse_T2,
        'rmse_T2': rmse_T2,
        'mae_T2': mae_T2,
        'dir_acc_T2': dir_acc_T2,
        'r2_T2': r2_T2,
        'mse_full': mse_full,
        'mae_full': mae_full,
        'dir_acc_full': dir_acc_full
    }

    return results


# ============================================================
# Visualization
# ============================================================
def plot_sample_predictions(predictions, ground_truth, num_examples=5, save_path='sample_predictions.png'):
    """Plot sample predictions vs ground truth."""
    num_examples = min(num_examples, len(predictions))

    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3*num_examples))

    if num_examples == 1:
        axes = [axes]

    for i in range(num_examples):
        pred = predictions[i]
        true = ground_truth[i]

        axes[i].plot(true, label='Ground Truth', linewidth=2, marker='o', markersize=4)
        axes[i].plot(pred, label='Prediction', linewidth=2, linestyle='--', marker='x', markersize=4)
        axes[i].axvline(x=T1-0.5, color='red', linestyle=':', linewidth=1.5, label=f'T1 (validation cutoff)')
        axes[i].set_title(f'Example {i+1}', fontsize=12)
        axes[i].set_xlabel('Timestep', fontsize=10)
        axes[i].set_ylabel('Close Price (normalized)', fontsize=10)
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Sample predictions saved to: {save_path}")
    plt.show()


# ============================================================
# Main Evaluation Script
# ============================================================
if __name__ == "__main__":
    print("="*70)
    print(f"{'EVALUATION WITH PREFIX VALIDATION':^70}")
    print("="*70)
    print(f"Selection metric: {SELECTION_METRIC}")
    print(f"Num samples per inference: {NUM_SAMPLES}")
    print(f"K-best to select: {K_BEST}")
    print(f"T1 (validation prefix): {T1}")
    print(f"T2 (final forecast): {T2}")
    print(f"Eval every {EVAL_EVERY_S} datapoints")
    print("="*70 + "\n")

    # ============================================================
    # Load Data
    # ============================================================
    print("Loading preprocessed data...")
    data = load_preprocessed_data(NPZ_PATH)

    X_test_batches = torch.from_numpy(data['X_test_batches']).float()
    Y_test_batches = torch.from_numpy(data['Y_test_batches']).float()

    print(f" Data loaded")
    print(f"  X_test: {X_test_batches.shape}")
    print(f"  Y_test: {Y_test_batches.shape}\n")

    NUM_FEATURES = X_test_batches.shape[-1]

    # Load column names
    col_names = pd.read_csv(COL_NAMES_PATH, header=None).squeeze().tolist()
    close_idx = col_names.index("log_return_close")
    print(f"Close price index: {close_idx}\n")

    # ============================================================
    # Load Model
    # ============================================================
    model, config = load_model(LOAD_MODEL_PATH, NUM_FEATURES, device)

    # ============================================================
    # Run Evaluation
    # ============================================================
    print("Starting evaluation with prefix validation and k-best selection...\n")

    results = evaluate_with_prefix_validation(
        model=model,
        X_test=X_test_batches,
        Y_test=Y_test_batches,
        close_idx=close_idx,
        num_samples=NUM_SAMPLES,
        k_best=K_BEST,
        eval_every_s=EVAL_EVERY_S,
        T1=T1,
        T2=T2,
        metric=SELECTION_METRIC,
        temperature=TEMPERATURE,
        device=device
    )

    # ============================================================
    # Print Results
    # ============================================================
    print("\n" + "="*70)
    print(f"{'EVALUATION RESULTS':^70}")
    print("="*70)

    print(f"\nT2 Period (Final Forecast, timesteps {T1}-{T1+T2-1}):")
    print(f"  MSE:  {results['mse_T2']:.6f}")
    print(f"  RMSE: {results['rmse_T2']:.6f}")
    print(f"  MAE:  {results['mae_T2']:.6f}")
    print(f"  Directional Accuracy: {results['dir_acc_T2']:.3f}")
    print(f"  R²: {results['r2_T2']:.3f}")

    print(f"\nFull Trajectory (timesteps 0-{DECODER_LENGTH_T-1}):")
    print(f"  MSE:  {results['mse_full']:.6f}")
    print(f"  MAE:  {results['mae_full']:.6f}")
    print(f"  Directional Accuracy: {results['dir_acc_full']:.3f}")

    print("\n" + "="*70 + "\n")

    # ============================================================
    # Save Results
    # ============================================================
    results_save_path = 'evaluation_results.npz'
    np.savez_compressed(
        results_save_path,
        predictions=results['predictions'],
        ground_truth=results['ground_truth'],
        mse_T2=results['mse_T2'],
        rmse_T2=results['rmse_T2'],
        mae_T2=results['mae_T2'],
        dir_acc_T2=results['dir_acc_T2'],
        r2_T2=results['r2_T2'],
        mse_full=results['mse_full'],
        mae_full=results['mae_full'],
        dir_acc_full=results['dir_acc_full']
    )
    print(f" Results saved to: {results_save_path}\n")

    # ============================================================
    # Visualize Sample Predictions
    # ============================================================
    # Extract close price predictions for visualization
    pred_close = results['predictions'][:, :, close_idx]
    true_close = results['ground_truth'][:, :, close_idx]

    plot_sample_predictions(pred_close, true_close, num_examples=5)

    print("\n=€ Evaluation complete!")
