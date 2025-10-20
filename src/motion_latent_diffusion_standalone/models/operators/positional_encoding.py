"""
Positional Encoding Layer for Transformers

ACTOR-style positional encoding that adds sinusoidal position information
directly to the input and applies dropout. This is used when pe_type="actor"
in the model configuration.

Different from position_encoding.py which returns position embeddings separately
(MLD-style), this module adds them directly to the input and includes dropout.
"""

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with dropout (ACTOR-style).

    Adds fixed sinusoidal position embeddings directly to the input features
    and applies dropout. This helps the model understand sequence order.

    The sinusoidal pattern uses different frequencies:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This creates unique, smooth position encodings that generalize to any length.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute sinusoidal positional encodings for all positions
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len, 1)

        # Frequency terms: decreasing exponentially
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        # Apply sine to even dimensions, cosine to odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape to (max_len, 1, d_model) for broadcasting
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as buffer (not a trainable parameter)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input and apply dropout.

        Args:
            x: Input features (seq_len, batch, d_model) or (batch, seq_len, d_model)

        Returns:
            Position-encoded features with dropout applied
        """
        if self.batch_first:
            # Add encoding for batch-first format
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            # Add encoding for sequence-first format
            x = x + self.pe[: x.shape[0], :]

        # Apply dropout for regularization
        return self.dropout(x)
