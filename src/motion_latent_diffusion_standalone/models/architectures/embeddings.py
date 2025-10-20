"""
Timestep Embedding Modules

Implements sinusoidal timestep embeddings for the diffusion denoiser.
As described in MLD paper Section 3.2, timesteps t are encoded using sinusoidal
position embeddings similar to Transformer architectures.
"""

import math
from typing import Literal
import torch
from torch import nn


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings for diffusion timesteps.

    Uses sinusoidal functions at different frequencies to encode the diffusion
    timestep t into a continuous embedding space. This allows the denoiser network
    to condition on which diffusion step it's at.

    Args:
        timesteps: 1-D tensor of N timestep indices, one per batch element
        embedding_dim: Dimension of the output embedding
        flip_sin_to_cos: If True, swap sin/cos order (matches diffusers convention)
        downscale_freq_shift: Shift applied to frequency calculation
        scale: Multiplicative scale applied to embeddings
        max_period: Maximum period (controls minimum frequency of embeddings)

    Returns:
        Sinusoidal embeddings of shape (N, embedding_dim)
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2

    # Compute frequency basis: exponentially decreasing frequencies
    # freq_i = 1 / (max_period^(2i/d)) for i in [0, half_dim)
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    # Apply frequencies to timesteps
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # Scale embeddings
    emb = scale * emb

    # Create embeddings with sin and cos at each frequency
    # Result: [sin(freq_0*t), ..., sin(freq_n*t), cos(freq_0*t), ..., cos(freq_n*t)]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # Optionally interleave sin and cos: [sin(f0*t), cos(f0*t), sin(f1*t), cos(f1*t), ...]
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # Pad with zeros if embedding_dim is odd
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb


class TimestepEmbedding(nn.Module):
    """
    Learned projection of timestep embeddings to latent dimension.

    Projects the sinusoidal timestep embedding through two linear layers with
    an activation in between. This allows the network to learn how to best
    incorporate timestep information into the denoising process.
    """

    def __init__(
        self,
        channel: int,
        time_embed_dim: int,
        act_fn: Literal["silu", "relu"] | None = "silu",
    ) -> None:
        super().__init__()

        # Two-layer MLP: channel -> time_embed_dim -> time_embed_dim
        self.linear_1 = nn.Linear(channel, time_embed_dim)
        if act_fn == "silu":
            self.act = nn.SiLU()  # Swish activation (smooth, non-monotonic)
        elif act_fn == "relu":
            self.act = nn.ReLU()
        else:
            self.act = None
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sample: Sinusoidal timestep embeddings (batch_size, channel)

        Returns:
            Projected embeddings (batch_size, time_embed_dim)
        """
        # First linear projection
        sample = self.linear_1(sample)

        # Non-linear activation
        if self.act is not None:
            sample = self.act(sample)

        # Second linear projection
        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Module):
    """
    Wrapper module for sinusoidal timestep embedding generation.

    Converts raw timestep indices to sinusoidal embeddings that can be
    processed by downstream networks.
    """

    def __init__(
        self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Raw timestep indices (batch_size,)

        Returns:
            Sinusoidal embeddings (batch_size, num_channels)
        """
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb
