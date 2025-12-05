# Crypto Price Forecasting with Deep Learning

This project implements and compares multiple deep learning architectures (Attention-based Seq2Seq, GRU, LSTM, MLP) for forecasting cryptocurrency prices (BTC/ETH). It provides a full pipeline from raw data acquisition and denoising, through feature engineering and normalization, to model training, evaluation, and profit/loss (PnL) backtesting.

## Project Structure & File Descriptions

### 1. Data Processing (`Data Processing/`)
Scripts in this folder handle **raw data fetching and generic preprocessing** that are reused across models (especially GRU/LSTM and the MLP baseline).

- **`01_Retrieve_API_Data.py`**
  - **Purpose:** Fetch historical OHLCV (Open, High, Low, Close, Volume) data from the Binance API.
  - **Key functions:** Initializes a Binance Spot client, iteratively downloads kline data in 1000‑candle chunks to respect rate limits, and constructs a unified `pandas` DataFrame.
  - **Outputs:**
    - Raw data CSVs such as `BTCUSDT_30m_10years.csv`, `ETHUSDT_30m_10years.csv`.

- **`02_Data_Preprocessing.py`**
  - **Purpose:** Turn raw CSVs into normalized, windowed sequences suitable for deep learning models.
  - **Key steps / functions:**
    - **Denoising:** Optionally applies Kalman filtering (and optionally wavelets) to OHLC price series to reduce high‑frequency noise.
    - **Feature engineering:** Adds Bollinger Band width/position, MACD, SMA and other technical indicators.
    - **Log returns:** Converts OHLC prices into log returns relative to the previous close, improving stationarity.
    - **Normalization:** Computes train‑set means/stds and applies Z‑score normalization consistently across train/val/test.
    - **Sequencing:** Builds sliding windows of length `SEQ_LEN` and future targets with horizon `HORIZON`.
    - **Splitting:** Splits sequences into Train / Validation / Test according to `train_frac`, `val_frac`, `test_frac`.
    - **Batching:** Packs sequences into fixed‑size batches (`batch_size`), storing the last (possibly smaller) batch separately when needed.
  - **Outputs:**
    - `preprocessed_data.npz` and variants  
      (`preprocessed_data_COMBINED.npz`, `preprocessed_data_SHUFFLED.npz`, etc.):  
      compressed arrays for
      - `X_train_batches`, `Y_train_batches`
      - `X_val_batches`, `Y_val_batches`
      - `X_test_batches`, `Y_test_batches`
      - plus timestamp batches in some configurations (e.g. for GRU PnL / assessment).
    - `mu_sigma_df.csv`: per‑feature mean and std used for normalization and later de‑normalization.
    - `column_names.csv`: ordered list of feature names corresponding to the columns in the arrays.
    - `datapoint_timestamps.csv`: timestamps aligned with the original datapoints, used for time‑alignment checks and backtesting.
    - `denoised_close_price_preview.png`: comparison plot of original vs denoised close prices over a random window.

> **Note:** The Attention model has its own preprocessing script in `Attention Model/data_processing.py` (see below) which reads the same CSVs but builds encoder–decoder style `(X, Y)` targets.

### 2. Attention Model (`Attention Model/`)
This folder contains an untested encoder–decoder model with attention and its own preprocessing/training/evaluation pipeline. This was purely for experimentation, and not part of the results for our project.

- **`data_processing.py`**
  - **Purpose:** Build **seq2seq‑style** datasets specifically for the Attention model.
  - **Key steps / functions:**
    - Loads `BTCUSDT_30m_10years.csv` (and optionally `ETHUSDT_30m_10years.csv`) from `Data Processing/`.
    - Applies feature engineering and denoising similar to `02_Data_Preprocessing.py` (Bollinger features, MACD, Kalman denoising).
    - Converts OHLC to log returns and drops unused columns.
    - Constructs encoder sequences of length `P = ENCODER_LENGTH_P` and decoder targets of length `T = HORIZON_T = T1 + T2`, where:
      - `T1` is the validation prefix length.
      - `T2` is the true forecast horizon.
    - Normalizes encoder and decoder data with train‑set statistics and batches them.
  - **Outputs:**
    - `preprocessed_data.npz` (and optional COMBINED/SHUFFLED variants) in `Attention Model/` containing
      - `X_train_batches`, `Y_train_batches`
      - `X_val_batches`, `Y_val_batches`
      - `X_test_batches`, `Y_test_batches`
    - `mu_sigma_df.csv` and `column_names.csv` tailored to this pipeline.

- **`model.py`**
  - **Purpose:** Define the Attention‑based Seq2Seq neural network architecture.
  - **Components:**
    - `Encoder`: LSTM/GRU that encodes past context `[batch_size, P, D]` into a sequence of hidden states.
    - `ScaledDotProductAttention`: Transformer‑style scaled dot‑product attention that uses the decoder hidden state as query and all encoder outputs as keys/values.
    - `Decoder`: Autoregressive decoder with
      - pre‑attention RNN over `[y_{t-1}, noise]`,
      - attention over encoder outputs,
      - post‑attention RNN over `[h_pre, context]`,
      - and linear heads that output per‑feature mean and std for a Gaussian predictive distribution.
    - `AttentionSeq2Seq`: Full encoder–decoder wrapper with a `forward` method that produces
      - `means`, `stds`, sampled trajectories, and attention weights.

- **`training.py`**
  - **Purpose:** Train the AttentionSeq2Seq model on the preprocessed seq2seq data.
  - **Key functions:**
    - Loads `preprocessed_data*.npz` and `column_names.csv` from `Attention Model/`.
    - Builds the model via `create_model`, configures AdamW optimizer and (optionally) a learning‑rate scheduler.
    - Uses **Gaussian negative log‑likelihood** (NLL) as the loss, with a lower bound on std to avoid collapse.
    - Implements **scheduled sampling** (teacher forcing probability decreasing after a few epochs).
    - Evaluates on the validation set each epoch, computing:
      - Average NLL,
      - directional accuracy on `log_return_close`,
      - R² score,
      - average predicted std.
    - Tracks training history and saves the best model checkpoint based on validation loss.
  - **Outputs:**
    - `attention_model_best.pth` and a final checkpoint with model state, optimizer state, config, and training history.
    - `training_curves.png`: multi‑panel plot of
      - Train vs Val NLL,
      - Validation directional accuracy,
      - Validation R²,
      - Average predicted std (for mode‑collapse monitoring).

- **`evaluation.py`**
  - **Purpose:** Perform advanced evaluation with **prefix validation + k‑best trajectory selection**.
  - **Key functions:**
    - Loads the best model checkpoint and test batches from `preprocessed_data*.npz`.
    - For selected test examples (every `EVAL_EVERY_S` samples):
      - Generates `NUM_SAMPLES` stochastic trajectories by sampling from the model’s predictive distribution.
      - Uses the first `T1` steps (validation prefix) of the true sequence to rank trajectories via
        - Dynamic Time Warping (DTW) distance, or
        - Pearson correlation (fallback if `fastdtw` is not installed).
      - Selects the top‑`K_BEST` trajectories and combines them via inverse‑error weighting to form a final prediction.
    - Computes metrics on both the forecast horizon (`T2`) and the full decoder window:
      - MSE, RMSE, MAE,
      - directional accuracy,
      - R² score.
  - **Outputs:**
    - `evaluation_results.npz`: predictions, ground truth, and summary metrics.
    - `sample_predictions.png`: plot of a few prediction vs ground‑truth sequences with the T1/T2 split indicated.

### 3. GRU Experimentation (`GRU_Experimentation/`)
This folder explores simpler recurrent architectures (GRU, LSTM, and an EncoderGRU variant) and integrates them with backtesting utilities.

- **`GRU.py`**
  - **Purpose:** Train GRU / LSTM / EncoderGRU models on the generic preprocessed data from `Data Processing/`.
  - **Key functions:**
    - Loads `preprocessed_data*.npz`, `column_names.csv`, and `mu_sigma_df.csv` from `../Data Processing/`.
    - Defines:
      - `GRU`: basic GRU that outputs a scalar per sample,
      - `EncoderGRU`: per‑timestep MLP encoder followed by a GRU,
      - `LSTM`: standard LSTM baseline.
    - Trains models to predict a **residual** over the last observed normalized close log‑return, improving stability.
    - After training, evaluates on Train/Val/Test with:
      - MAE, RMSE, and sign accuracy (directional accuracy),
      - comparison to simple baselines (persistence and zero‑return).
  - **Outputs:**
    - `cur_best_model.pth`: checkpoint with model + optimizer + scheduler states and final losses.
    - `loss_curve.png`: training vs validation loss across epochs.

- **`Assess_Predictions.py`**
  - **Purpose:** Interactive inspection of GRU family predictions on the test set.
  - **Key functions:**
    - Loads `cur_best_model.pth` and the corresponding test batches plus timestamp batches.
    - Iterates through test sequences, printing:
      - the tail of the input close‑price sequence,
      - the (de‑normalized) true and predicted future values,
      - running directional accuracy statistics.
    - Verifies strict time alignment: target timestamps must be after the last input timestamp, and de‑normalized Y must match a future X point offset by the horizon.
  - **Outputs:** Console‑based interactive summaries (no additional files written).

- **`PnL.py`**
  - **Purpose:** Backtest a simple long/short strategy driven by EncoderGRU predictions.
  - **Key functions:**
    - Loads the EncoderGRU checkpoint, feature normalization stats, `datapoint_timestamps.csv`, and raw price CSV (`BTCUSDT_30m_10years.csv`).
    - Automatically infers how the targets were scaled/normalized by comparing reconstructed prices to the raw price path.
    - Simulates trades every `HORIZON` steps:
      - takes the sign of predicted horizon log‑returns as long/short direction,
      - applies configurable slippage and fees in basis points,
      - updates portfolio equity over time.
  - **Outputs:**
    - `trades_log.csv`: per‑trade direction, entry/exit prices, raw and net returns, and balance.
    - `equity_curve.csv`: resulting equity curve over the backtest.
    - An equity‑curve plot (displayed via Matplotlib; can be saved/modified as needed).

### 3.1 Directional Loss Experiment (`Directional_Loss_Experiment/`)

This folder contains an auxiliary experiment that modifies the GRU training objective to better align with trading performance by explicitly penalizing direction errors.

- **`train_directional.py`**
  - **Purpose:** Train an `EncoderGRU` model using a **direction-weighted MSE** loss that multiplies the squared error by a factor \(1 + \alpha\) whenever the predicted sign of the return disagrees with the true sign.
  - **Key details:**
    - Uses the same preprocessed data as `GRU_Experimentation/GRU.py` (from `Data Processing/preprocessed_data.npz`).
    - Model architecture: per‑timestep MLP encoder \(\rightarrow\) GRU \(\rightarrow\) LayerNorm \(\rightarrow\) linear head predicting a residual over the last normalized close log‑return.
    - Loss: standard MSE when `sign(pred) == sign(target)`, and \((1+\alpha)\times\) MSE when the signs disagree (with \(\alpha = 10\) in our final run).
  - **Result:** Achieves train/validation/test directional accuracies of **55.18% / 53.32% / 53.85%**, compared to \(\approx 50\%\) under a vanilla MSE objective, showing that loss design alone can yield a meaningful 3–4 percentage point gain in short‑horizon directional forecasting.

- **`check_results.py`**
  - **Purpose:** Convenience script to reload `directional_model.pth`, recompute train/val/test MSE and directional accuracy, and print a concise summary (used to generate the 55.18% / 53.32% / 53.85% figures).

- **`direction_acc_curve.png`**
  - **Purpose:** Visualization of validation directional accuracy over epochs for the direction‑weighted loss experiment, with a horizontal reference line at 50% to highlight the improvement over random guessing.

### 4. MLP Baseline & Artifacts

- **`Skeleton_Neural_Net_Template.py`**
  - **Purpose:** Provide a simple flattened‑sequence MLP baseline for future log‑return prediction.
  - **Key functions:**
    - Loads `preprocessed_data.npz` from `Data Processing/`, merges batches and last batches into continuous sets.
    - Uses sliding windows of length `W` and feature dimension `F` and flattens them into vectors of length `W × F`.
    - Trains a configurable MLP with dropout using MSE loss on scalar log‑returns.
    - Evaluates against a zero‑return baseline in terms of MSE, MAE, RMSE, R², correlation, directional accuracy, and a small synthetic PnL.
  - **Outputs (primarily in `artifacts/` at the project root):**
    - `learning_curves_loss_plot.jpg`: training/validation/test loss curves.
    - `test_scatter.png`: predicted vs true log‑returns scatter plot.
    - `test_err_hist.png`: prediction error histogram.
    - `test_pnl.png`: equity curve for a tiny long/short PnL strategy based on the MLP.
    - `test_preds.csv`: `(y_true_logret, y_pred_logret)` pairs for downstream analysis.
    - `exp_results.pkl`: serialized experiment histories (e.g., loss curves) for quick comparison across runs.

### 5. Other Top‑Level Items

- **`artifacts/` (root folder)**  
  Central location for plots, CSVs, and other outputs from baseline experiments (and can be reused for additional analysis).

- **`AWS/`**  
  Reserved for future cloud (AWS) deployment or training scripts; currently empty in this repository snapshot.

---

## How to Run

### Prerequisites
Install the required Python packages, for example via `pip`:

```bash
pip install torch numpy pandas matplotlib binance-connector pandas_ta scipy fastdtw
```

You may also need to install any additional packages referenced in the scripts (e.g., `tqdm` for progress bars).

### Step 1: Prepare Raw & Generic Preprocessed Data

From the project root:

```bash
cd "Data Processing"
```

1. **(Optional) Fetch new raw data** from Binance (requires API keys configured in `01_Retrieve_API_Data.py`):

   ```bash
   python 01_Retrieve_API_Data.py
   ```

   This produces raw CSVs such as `BTCUSDT_30m_10years.csv`.

2. **Run the generic preprocessing pipeline** used by GRU/LSTM and the MLP baseline:

   ```bash
   python 02_Data_Preprocessing.py
   ```

   This generates:
   - `preprocessed_data*.npz` in `Data Processing/`
   - `mu_sigma_df.csv`, `column_names.csv`, `datapoint_timestamps.csv`
   - `denoised_close_price_preview.png` for visual inspection.

### Step 2: Train & Evaluate the Attention Model

The Attention model uses its **own** preprocessing script in `Attention Model/` that reads the raw CSVs from `Data Processing/`.

```bash
cd "../Attention Model"
```

1. **Build encoder–decoder datasets for Attention:**

   ```bash
   python data_processing.py
   ```

   This creates `preprocessed_data*.npz` local to `Attention Model/`.

2. **Train the Attention Seq2Seq model:**

   ```bash
   python training.py
   ```

   This will save `attention_model_best.pth` and `training_curves.png`.

3. **Run advanced evaluation with prefix validation and k‑best selection:**

   ```bash
   python evaluation.py
   ```

   This produces `evaluation_results.npz` and `sample_predictions.png`.

### Step 3: Train & Backtest GRU/LSTM Models

From the project root:

```bash
cd "GRU_Experimentation"
```

1. **Train GRU / LSTM / EncoderGRU** using the generic preprocessed data:

   ```bash
   python GRU.py
   ```

   This writes `cur_best_model.pth` and `loss_curve.png`.

2. **Interactively inspect predictions on the test set:**

   ```bash
   python Assess_Predictions.py
   ```

   Use the prompts to step through sequences and see true vs predicted values and running directional accuracy.

3. **Run PnL backtest for the EncoderGRU strategy:**

   ```bash
   python PnL.py
   ```

   This creates `trades_log.csv` and `equity_curve.csv` and displays an equity‑curve plot.

### Step 4: Run the MLP Baseline

From the project root, after `preprocessed_data.npz` has been created by `02_Data_Preprocessing.py`:

```bash
python Skeleton_Neural_Net_Template.py
```

This will train the MLP baseline, evaluate it on the test set, and write plots and CSVs into the `artifacts/` folder.

