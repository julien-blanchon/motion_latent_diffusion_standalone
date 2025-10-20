"""
T2M Motion Encoder Architectures

These encoders are used for motion understanding and discrimination.
They were originally trained for motion retrieval (text-to-motion matching)
and provide better motion embeddings for classification than VAE latents.

Architecture:
1. MovementConvEncoder: Extracts local motion patterns using 1D convolutions
2. MotionEncoderBiGRUCo: Captures temporal structure using bidirectional GRU

These encoders are used in the evaluation metrics (R-Precision, FID, etc.)
and provide much better discrimination between motion types than VAE latents.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from huggingface_hub import PyTorchModelHubMixin


class MovementConvEncoder(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/julien-blanchon/motion_latent_diffusion_standalone",
):
    """
    Movement encoder using 1D convolutions.

    Extracts local motion features from input motion sequences.
    Uses strided convolutions to downsample temporally.

    Args:
        input_size: Input feature dimension (e.g., 259 for motion features - 4)
        hidden_size: Hidden layer dimension (e.g., 512)
        output_size: Output feature dimension (e.g., 512)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        # Two strided convolutions with dropout and leaky ReLU (downsamples by 4x)
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, seq_len, input_size)

        Returns:
            outputs: (batch_size, seq_len//4, output_size)
        """
        # Rearrange to channel-first format for convolutions
        inputs = inputs.permute(0, 2, 1)  # (B, C, T)
        outputs = self.main(inputs).permute(0, 2, 1)  # (B, T', C)
        return self.out_net(outputs)


class MotionEncoderBiGRUCo(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/julien-blanchon/motion_latent_diffusion_standalone",
):
    """
    Motion encoder using bidirectional GRU.

    Captures temporal dependencies in motion sequences and produces
    a single embedding vector per sequence.

    Args:
        input_size: Input feature dimension (e.g., 512 from movement encoder)
        hidden_size: GRU hidden dimension (e.g., 1024)
        output_size: Output embedding dimension (e.g., 512)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()

        self.input_emb = nn.Linear(input_size, hidden_size)
        # Bidirectional GRU to capture temporal context from both directions
        self.gru = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        # Project concatenated bidirectional outputs to final embedding
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        self.hidden_size = hidden_size
        # Learnable initial hidden state for both directions
        self.hidden = nn.Parameter(
            torch.randn((2, 1, self.hidden_size), requires_grad=True)
        )

    def forward(self, inputs: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, seq_len, input_size) - movement features
            m_lens: (batch_size,) - actual sequence lengths (for packing)

        Returns:
            embeddings: (batch_size, output_size) - motion embeddings
        """
        num_samples = inputs.shape[0]

        # Project inputs to hidden size
        input_embs = self.input_emb(inputs)

        # Replicate hidden state across batch dimension
        hidden = self.hidden.repeat(1, num_samples, 1)

        # Pack variable-length sequences for efficient GRU processing
        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        # Run bidirectional GRU
        gru_seq, gru_last = self.gru(emb, hidden)

        # Concatenate forward and backward final hidden states
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        # Project to output dimension
        return self.output_net(gru_last)
