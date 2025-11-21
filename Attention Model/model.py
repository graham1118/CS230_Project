# Model architecture file for Attention-based Sequence-to-Sequence forecasting

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# Configuration (must match training.py and data_processing.py)
# ============================================================
MODEL_TYPE = 'LSTM'  # 'LSTM' or 'GRU'
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LATENT_SIZE = 16  # Dimension of random vector injected at each decoder timestep
USE_ATTENTION = True
ATTENTION_TEMPERATURE = 1.0
ENCODER_LENGTH_P = 32
DECODER_LENGTH_T = 16  # T = T1 + T2
T1 = 6
T2 = 10
DROPOUT = 0.2
MIN_STD_CLAMP = 1e-3  # Minimum std to prevent collapse


# ============================================================
# Encoder Module
# ============================================================
class Encoder(nn.Module):
    """
    Causal sequence encoder using LSTM or GRU.

    Processes past P timesteps and produces hidden states for attention.
    Supports both LSTM and GRU architectures.
    """

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, model_type='LSTM'):
        """
        Parameters
        ----------
        input_size : int
            Number of input features (D)
        hidden_size : int
            Hidden dimension
        num_layers : int
            Number of stacked RNN layers
        dropout : float
            Dropout rate applied after each layer
        model_type : str
            'LSTM' or 'GRU'
        """
        super(Encoder, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, P, D]

        Returns
        -------
        encoder_outputs : torch.Tensor
            All hidden states, shape [batch_size, P, hidden_size]
        hidden_state : tuple or torch.Tensor
            Final hidden state(s) - tuple (h, c) for LSTM or tensor h for GRU
        """
        # x: [batch_size, P, D]

        # Process through RNN
        encoder_outputs, hidden_state = self.rnn(x)
        # encoder_outputs: [batch_size, P, hidden_size]
        # hidden_state: ([num_layers, batch_size, hidden_size], [num_layers, batch_size, hidden_size]) for LSTM
        #               or [num_layers, batch_size, hidden_size] for GRU

        # Apply dropout to encoder outputs
        encoder_outputs = self.dropout(encoder_outputs)

        return encoder_outputs, hidden_state


# ============================================================
# Attention Module
# ============================================================
class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism.

    Standard transformer-style attention with temperature scaling.
    Query comes from decoder hidden state, keys/values from encoder outputs.
    """

    def __init__(self, hidden_size, temperature=1.0, dropout=0.2):
        """
        Parameters
        ----------
        hidden_size : int
            Dimension of hidden states
        temperature : float
            Temperature for softmax scaling
        dropout : float
            Dropout rate applied to attention weights
        """
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

        # Scale factor for dot-product
        self.scale = np.sqrt(hidden_size * temperature)

    def forward(self, query, keys, values, mask=None):
        """
        Compute attention and context vector.

        Parameters
        ----------
        query : torch.Tensor
            Query from decoder, shape [batch_size, hidden_size]
        keys : torch.Tensor
            Keys from encoder, shape [batch_size, P, hidden_size]
        values : torch.Tensor
            Values from encoder, shape [batch_size, P, hidden_size]
        mask : torch.Tensor, optional
            Mask for attention scores, shape [batch_size, P]

        Returns
        -------
        context : torch.Tensor
            Context vector, shape [batch_size, hidden_size]
        attention_weights : torch.Tensor
            Attention weights, shape [batch_size, P]
        """
        # query: [batch_size, hidden_size]
        # keys: [batch_size, P, hidden_size]
        # values: [batch_size, P, hidden_size]

        # Expand query to match keys dimension for batch matrix multiplication
        query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # Compute attention scores: Q @ K^T
        # scores: [batch_size, 1, P]
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale

        # Apply mask if provided (for padding, though not needed in our case)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, 1, P]

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Compute context vector: attention_weights @ V
        context = torch.bmm(attention_weights, values)  # [batch_size, 1, hidden_size]
        context = context.squeeze(1)  # [batch_size, hidden_size]
        attention_weights = attention_weights.squeeze(1)  # [batch_size, P]

        return context, attention_weights


# ============================================================
# Decoder Module
# ============================================================
class Decoder(nn.Module):
    """
    Autoregressive decoder with pre-attention LSTM ’ attention ’ post-attention LSTM.

    Architecture at each timestep t:
    1. Concatenate [y_{t-1}, injected_noise] ’ decoder_input
    2. Pre-attention LSTM processes decoder_input ’ h_pre
    3. Attention over encoder outputs using h_pre as query ’ context
    4. Concatenate [h_pre, context]
    5. Post-attention LSTM processes [h_pre, context] ’ h_post
    6. Final layer maps h_post ’ (mean, std) for all D features
    """

    def __init__(self, input_size, hidden_size, latent_size, num_layers=2,
                 dropout=0.2, attention_temperature=1.0, min_std_clamp=1e-3, model_type='LSTM'):
        """
        Parameters
        ----------
        input_size : int
            Number of features (D) in decoder output
        hidden_size : int
            Hidden dimension
        latent_size : int
            Dimension of injected random noise
        num_layers : int
            Number of layers in pre- and post-attention RNNs
        dropout : float
            Dropout rate
        attention_temperature : float
            Temperature for attention mechanism
        min_std_clamp : float
            Minimum std value to prevent collapse
        model_type : str
            'LSTM' or 'GRU'
        """
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.min_std_clamp = min_std_clamp
        self.model_type = model_type

        # Pre-attention LSTM/GRU: processes [y_{t-1}, noise]
        if model_type == 'LSTM':
            self.pre_attention_rnn = nn.LSTM(
                input_size=input_size + latent_size,  # y_{t-1} + noise
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif model_type == 'GRU':
            self.pre_attention_rnn = nn.GRU(
                input_size=input_size + latent_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Attention mechanism
        self.attention = ScaledDotProductAttention(
            hidden_size=hidden_size,
            temperature=attention_temperature,
            dropout=dropout
        )

        # Post-attention LSTM/GRU: processes [h_pre, context]
        if model_type == 'LSTM':
            self.post_attention_rnn = nn.LSTM(
                input_size=hidden_size * 2,  # h_pre + context
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif model_type == 'GRU':
            self.post_attention_rnn = nn.GRU(
                input_size=hidden_size * 2,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )

        # Dropout layers
        self.dropout = nn.Dropout(dropout)

        # Final projection to mean and std
        self.fc_mean = nn.Linear(hidden_size, input_size)
        self.fc_std = nn.Linear(hidden_size, input_size)

    def init_hidden(self, batch_size, device):
        """
        Initialize hidden states for pre- and post-attention RNNs.

        Returns
        -------
        pre_hidden : tuple or torch.Tensor
            Hidden state for pre-attention RNN
        post_hidden : tuple or torch.Tensor
            Hidden state for post-attention RNN
        """
        if self.model_type == 'LSTM':
            # LSTM: return (h, c)
            pre_h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            pre_c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            post_h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            post_c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return (pre_h, pre_c), (post_h, post_c)
        else:
            # GRU: return h only
            pre_h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            post_h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return pre_h, post_h

    def forward_step(self, y_prev, injected_noise, encoder_outputs,
                     pre_hidden, post_hidden):
        """
        Single decoder timestep.

        Parameters
        ----------
        y_prev : torch.Tensor
            Previous output, shape [batch_size, D]
        injected_noise : torch.Tensor
            Random noise, shape [batch_size, latent_size]
        encoder_outputs : torch.Tensor
            Encoder outputs for attention, shape [batch_size, P, hidden_size]
        pre_hidden : tuple or torch.Tensor
            Hidden state for pre-attention RNN
        post_hidden : tuple or torch.Tensor
            Hidden state for post-attention RNN

        Returns
        -------
        mean : torch.Tensor
            Predicted mean, shape [batch_size, D]
        std : torch.Tensor
            Predicted std, shape [batch_size, D]
        pre_hidden : tuple or torch.Tensor
            Updated pre-attention hidden state
        post_hidden : tuple or torch.Tensor
            Updated post-attention hidden state
        attention_weights : torch.Tensor
            Attention weights, shape [batch_size, P]
        """
        batch_size = y_prev.size(0)

        # Step 1: Concatenate [y_{t-1}, noise]
        decoder_input = torch.cat([y_prev, injected_noise], dim=-1)  # [batch_size, D + latent_size]
        decoder_input = decoder_input.unsqueeze(1)  # [batch_size, 1, D + latent_size]

        # Step 2: Pre-attention RNN
        h_pre_out, pre_hidden = self.pre_attention_rnn(decoder_input, pre_hidden)
        # h_pre_out: [batch_size, 1, hidden_size]
        h_pre = h_pre_out.squeeze(1)  # [batch_size, hidden_size]

        # Step 3: Apply dropout
        h_pre = self.dropout(h_pre)

        # Step 4: Attention (query = h_pre, keys/values = encoder_outputs)
        # All encoder outputs are accessible since they represent past data
        context, attention_weights = self.attention(h_pre, encoder_outputs, encoder_outputs)
        # context: [batch_size, hidden_size]

        # Step 5: Apply dropout to context
        context = self.dropout(context)

        # Step 6: Concatenate [h_pre, context]
        post_input = torch.cat([h_pre, context], dim=-1)  # [batch_size, 2*hidden_size]
        post_input = post_input.unsqueeze(1)  # [batch_size, 1, 2*hidden_size]

        # Step 7: Post-attention RNN
        h_post_out, post_hidden = self.post_attention_rnn(post_input, post_hidden)
        # h_post_out: [batch_size, 1, hidden_size]
        h_post = h_post_out.squeeze(1)  # [batch_size, hidden_size]

        # Step 8: Apply dropout
        h_post = self.dropout(h_post)

        # Step 9: Final projection to mean and std
        mean = self.fc_mean(h_post)  # [batch_size, D]
        raw_std = self.fc_std(h_post)  # [batch_size, D]

        # Apply softplus + clamp to ensure positive std
        std = F.softplus(raw_std) + self.min_std_clamp  # [batch_size, D]

        return mean, std, pre_hidden, post_hidden, attention_weights

    def forward(self, encoder_outputs, start_token, T, temperature=1.0,
                ground_truth=None, teacher_forcing_prob=0.0):
        """
        Full decoder forward pass for training or inference.

        Parameters
        ----------
        encoder_outputs : torch.Tensor
            Encoder outputs, shape [batch_size, P, hidden_size]
        start_token : torch.Tensor
            Initial input (last encoder input), shape [batch_size, D]
        T : int
            Number of timesteps to decode
        temperature : float
            Sampling temperature
        ground_truth : torch.Tensor, optional
            Ground truth for scheduled sampling, shape [batch_size, T, D]
        teacher_forcing_prob : float
            Probability of using ground truth instead of sampled output

        Returns
        -------
        means : torch.Tensor
            Predicted means for all timesteps, shape [batch_size, T, D]
        stds : torch.Tensor
            Predicted stds for all timesteps, shape [batch_size, T, D]
        samples : torch.Tensor
            Sampled outputs, shape [batch_size, T, D]
        all_attention_weights : torch.Tensor
            Attention weights for all timesteps, shape [batch_size, T, P]
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        # Initialize hidden states
        pre_hidden, post_hidden = self.init_hidden(batch_size, device)

        # Storage for outputs
        means = []
        stds = []
        samples = []
        all_attention_weights = []

        # Initialize with start token
        y_prev = start_token

        # Autoregressive loop
        for t in range(T):
            # Sample random noise for this timestep
            injected_noise = torch.randn(batch_size, self.latent_size, device=device)

            # Forward through one timestep
            mean_t, std_t, pre_hidden, post_hidden, attn_weights = self.forward_step(
                y_prev, injected_noise, encoder_outputs, pre_hidden, post_hidden
            )

            # Sample from distribution
            epsilon = torch.randn_like(mean_t)
            y_t = mean_t + std_t * epsilon * temperature

            # Store outputs
            means.append(mean_t)
            stds.append(std_t)
            samples.append(y_t)
            all_attention_weights.append(attn_weights)

            # Scheduled sampling: with probability teacher_forcing_prob, use ground truth
            if ground_truth is not None and teacher_forcing_prob > 0.0:
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_prob
                if use_teacher_forcing:
                    y_prev = ground_truth[:, t, :]
                else:
                    y_prev = y_t
            else:
                y_prev = y_t

        # Stack outputs
        means = torch.stack(means, dim=1)  # [batch_size, T, D]
        stds = torch.stack(stds, dim=1)    # [batch_size, T, D]
        samples = torch.stack(samples, dim=1)  # [batch_size, T, D]
        all_attention_weights = torch.stack(all_attention_weights, dim=1)  # [batch_size, T, P]

        return means, stds, samples, all_attention_weights


# ============================================================
# Full Sequence-to-Sequence Model
# ============================================================
class AttentionSeq2Seq(nn.Module):
    """
    Complete encoder-decoder model with attention.

    Combines Encoder, Attention, and Decoder into a single model.
    """

    def __init__(self, input_size, hidden_size, latent_size, num_layers=2,
                 dropout=0.2, attention_temperature=1.0, min_std_clamp=1e-3, model_type='LSTM'):
        """
        Parameters
        ----------
        input_size : int
            Number of features (D)
        hidden_size : int
            Hidden dimension
        latent_size : int
            Dimension of injected noise
        num_layers : int
            Number of RNN layers
        dropout : float
            Dropout rate
        attention_temperature : float
            Attention temperature
        min_std_clamp : float
            Minimum std value
        model_type : str
            'LSTM' or 'GRU'
        """
        super(AttentionSeq2Seq, self).__init__()

        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type
        )

        self.decoder = Decoder(
            input_size=input_size,
            hidden_size=hidden_size,
            latent_size=latent_size,
            num_layers=num_layers,
            dropout=dropout,
            attention_temperature=attention_temperature,
            min_std_clamp=min_std_clamp,
            model_type=model_type
        )

    def forward(self, x, T, temperature=1.0, ground_truth=None, teacher_forcing_prob=0.0):
        """
        Full forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Encoder input, shape [batch_size, P, D]
        T : int
            Number of decoder timesteps
        temperature : float
            Sampling temperature
        ground_truth : torch.Tensor, optional
            Ground truth for training, shape [batch_size, T, D]
        teacher_forcing_prob : float
            Scheduled sampling probability

        Returns
        -------
        means : torch.Tensor
            Predicted means, shape [batch_size, T, D]
        stds : torch.Tensor
            Predicted stds, shape [batch_size, T, D]
        samples : torch.Tensor
            Sampled trajectories, shape [batch_size, T, D]
        attention_weights : torch.Tensor
            Attention weights, shape [batch_size, T, P]
        """
        # Encode
        encoder_outputs, _ = self.encoder(x)

        # Start token: last timestep of encoder input
        start_token = x[:, -1, :]

        # Decode
        means, stds, samples, attention_weights = self.decoder(
            encoder_outputs=encoder_outputs,
            start_token=start_token,
            T=T,
            temperature=temperature,
            ground_truth=ground_truth,
            teacher_forcing_prob=teacher_forcing_prob
        )

        return means, stds, samples, attention_weights


# ============================================================
# Helper function to create model
# ============================================================
def create_model(input_size, hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE,
                num_layers=NUM_LAYERS, dropout=DROPOUT,
                attention_temperature=ATTENTION_TEMPERATURE,
                min_std_clamp=MIN_STD_CLAMP, model_type=MODEL_TYPE):
    """
    Factory function to create the Seq2Seq model.

    Parameters
    ----------
    input_size : int
        Number of input features (D)
    hidden_size : int
        Hidden dimension
    latent_size : int
        Latent noise dimension
    num_layers : int
        Number of RNN layers
    dropout : float
        Dropout rate
    attention_temperature : float
        Attention temperature
    min_std_clamp : float
        Minimum std clamp
    model_type : str
        'LSTM' or 'GRU'

    Returns
    -------
    model : AttentionSeq2Seq
        Initialized model
    """
    model = AttentionSeq2Seq(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
        dropout=dropout,
        attention_temperature=attention_temperature,
        min_std_clamp=min_std_clamp,
        model_type=model_type
    )
    return model


if __name__ == "__main__":
    # Test model architecture
    print("Testing model architecture...")

    # Parameters
    batch_size = 32
    P = ENCODER_LENGTH_P  # 32
    T = DECODER_LENGTH_T  # 16
    D = 10  # Number of features

    # Create model
    model = create_model(input_size=D)

    # Create dummy data
    x = torch.randn(batch_size, P, D)
    y_true = torch.randn(batch_size, T, D)

    print(f"\nInput shape: {x.shape}")
    print(f"Ground truth shape: {y_true.shape}")

    # Forward pass
    means, stds, samples, attention_weights = model(
        x, T=T, temperature=1.0, ground_truth=y_true, teacher_forcing_prob=0.5
    )

    print(f"\nOutput shapes:")
    print(f"  Means: {means.shape}")
    print(f"  Stds: {stds.shape}")
    print(f"  Samples: {samples.shape}")
    print(f"  Attention weights: {attention_weights.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n Model architecture test passed!")
