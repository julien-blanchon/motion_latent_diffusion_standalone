"""
Position Encodings for Transformers

Provides position information to transformer models, which are otherwise
position-agnostic. Two variants are supported:

1. Sinusoidal (PositionEmbeddingSine1D): Fixed sinusoidal patterns at different
   frequencies. No learnable parameters, generalizes to any sequence length.

2. Learned (PositionEmbeddingLearned1D): Learnable embeddings for each position.
   More flexible but limited to max_len positions.

Adapted from DETR (Facebook Research).
"""

from typing import Literal
import numpy as np
import torch
from torch import Tensor, nn


class PositionEmbeddingSine1D(nn.Module):
    """
    Fixed sinusoidal position embeddings for 1D sequences.

    Uses sine and cosine functions at different frequencies to create
    unique embeddings for each position. Same idea as in "Attention is All You Need".
    """

    def __init__(
        self, d_model: int, max_len: int = 500, batch_first: bool = False
    ) -> None:
        super().__init__()
        self.batch_first = batch_first

        # Pre-compute positional encodings for all positions up to max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len, 1)

        # Compute frequency terms: 1 / (10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # PE(pos, 2i) = sin(pos/10000^(2i/d))
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # PE(pos, 2i+1) = cos(pos/10000^(2i/d))

        # Reshape for sequence-first: (max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as buffer (saved with model but not trained)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (determines sequence length)

        Returns:
            Position embeddings matching input shape
        """
        if self.batch_first:
            # Return (1, seq_len, d_model) for broadcasting
            pos = self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            # Return (seq_len, 1, d_model) for broadcasting
            pos = self.pe[: x.shape[0], :]
        return pos


class PositionEmbeddingLearned1D(nn.Module):
    """
    Learnable position embeddings for 1D sequences.

    Unlike sinusoidal embeddings, these are learned during training.
    Can potentially better capture task-specific positional patterns,
    but limited to sequences up to max_len.
    """

    def __init__(
        self, d_model: int, max_len: int = 500, batch_first: bool = False
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        # Learnable embedding for each position up to max_len
        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize with uniform distribution"""
        nn.init.uniform_(self.pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (determines sequence length)

        Returns:
            Position-encoded tensor
        """
        if self.batch_first:
            # Return position embeddings for batch-first format (separate)
            return self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            # Add position embeddings directly (sequence-first)
            return x + self.pe[: x.shape[0], :]


def build_position_encoding(
    N_steps: int,
    position_embedding: Literal["sine", "learned", "v2", "v3"] = "sine",
    embedding_dim: Literal["1D", "2D"] = "1D",
) -> nn.Module:
    """
    Build position encoding module.

    Args:
        N_steps: Dimension of the encoding
        position_embedding: Type of embedding ('sine', 'learned', 'v2', or 'v3')
        embedding_dim: Dimension type ('1D' or '2D')

    Returns:
        Position embedding module
    """
    if embedding_dim == "1D":
        if position_embedding in ("v2", "sine"):
            # Sinusoidal encoding (DETR default)
            position_embedding_module = PositionEmbeddingSine1D(N_steps)
        elif position_embedding in ("v3", "learned"):
            # Learnable encoding
            position_embedding_module = PositionEmbeddingLearned1D(N_steps)
        else:
            raise ValueError(f"not supported {position_embedding}")
    else:
        raise ValueError(f"not supported {embedding_dim}")

    return position_embedding_module
