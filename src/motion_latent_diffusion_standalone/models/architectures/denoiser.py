"""
Motion Latent Diffusion Denoiser (ε_θ)

Implements the denoiser network for the diffusion model, as described in MLD paper
Section 3.2. The denoiser predicts the noise added to motion latents at each diffusion
timestep, conditioned on text embeddings.

Architecture (from paper Table 13):
- 9 transformer layers with 4 heads, 256 hidden dim, 1024 FFN dim
- Supports both encoder-only (trans_enc) and decoder (trans_dec) architectures
- Can use U-Net style skip connections for better gradient flow
- Conditioning via concatenation of timestep and text embeddings
"""

import torch
import torch.nn as nn
from typing import Literal
from huggingface_hub import PyTorchModelHubMixin

from .embeddings import TimestepEmbedding, Timesteps
from ..operators import PositionalEncoding
from ..operators.cross_attention import (
    SkipTransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from ..operators.position_encoding import build_position_encoding
from ..utils import lengths_to_mask


class Denoiser(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/julien-blanchon/motion_latent_diffusion_standalone",
):
    """
    Motion Latent Diffusion Denoiser Network.

    The core denoising network ε_θ(z_t, t, c) that predicts the noise added to
    latent motion z_t at timestep t, conditioned on text/action embedding c.

    MLD uses a transformer architecture (encoder or decoder) to process the noisy
    latent along with timestep and condition information. The model is trained to
    predict noise, which is then used to iteratively denoise samples during generation.

    Key features:
    - Supports text and action conditioning via cross-attention or concatenation
    - Optional U-Net style skip connections for better information flow
    - Classifier-free guidance support for controllable generation
    - Flexible architecture: encoder-only (faster) or decoder (cross-attention)
    """

    def __init__(
        self,
        nfeats: int = 263,  # Motion feature dimension (unused if working in latent space)
        condition: Literal["text", "text_uncond", "action"] = "text",
        latent_dim: list[int] = [
            1,
            256,
        ],  # [sequence_length, hidden_dim] in latent space
        ff_size: int = 1024,  # Feedforward layer dimension
        num_layers: int = 6,  # Number of transformer layers (must be odd for skip connections)
        num_heads: int = 4,  # Number of attention heads
        dropout: float = 0.1,
        normalize_before: bool = False,  # Pre-norm vs post-norm transformer
        activation: Literal["gelu", "relu"] = "gelu",
        flip_sin_to_cos: bool = True,
        return_intermediate_dec: bool = False,
        position_embedding: Literal["sine", "learned", "v2", "v3"] = "learned",
        arch: Literal["trans_enc", "trans_dec"] = "trans_enc",  # Architecture type
        freq_shift: int = 0,
        guidance_scale: float = 7.5,  # Classifier-free guidance scale
        guidance_uncondp: float = 0.1,  # Probability of unconditional training
        text_encoded_dim: int = 768,  # CLIP text embedding dimension
        nclasses: int = 10,  # Number of action classes (for action conditioning)
        skip_connect: bool = True,  # Enable U-Net style skip connections
        pe_type: Literal["actor", "mld"] = "mld",  # Position encoding style
        **kwargs,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim[-1]  # Hidden dimension (e.g., 256)
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.abl_plus = False  # Ablation: add instead of concat embeddings
        self.ablation_skip_connection = skip_connect
        self.diffusion_only = False  # Always False for MLD (uses VAE latents)
        self.arch = arch
        self.pe_type = pe_type

        # Diffusion-only mode: work directly on raw features (not used in MLD)
        # This mode bypasses the VAE and applies diffusion directly to motion features
        if self.diffusion_only:
            self.pose_embd = nn.Linear(nfeats, self.latent_dim)
            self.pose_proj = nn.Linear(self.latent_dim, nfeats)

        # === Timestep and Condition Embedding Setup ===
        # The denoiser needs to know both "when" (timestep) and "what" (condition)

        if self.condition in ["text", "text_uncond"]:
            # Text-conditioned diffusion: condition on CLIP embeddings
            # Timestep embeddings match CLIP dimension for easier fusion
            self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos, freq_shift)
            self.time_embedding = TimestepEmbedding(text_encoded_dim, self.latent_dim)

            # Project text embeddings to latent_dim if needed for dimension matching
            if text_encoded_dim != self.latent_dim:
                self.emb_proj = nn.Sequential(
                    nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim)
                )
        elif self.condition == "action":
            # Action-conditioned diffusion: condition on action class labels
            self.time_proj = Timesteps(self.latent_dim, flip_sin_to_cos, freq_shift)
            self.time_embedding = TimestepEmbedding(self.latent_dim, self.latent_dim)
            # Learnable action embeddings with classifier-free guidance
            self.emb_proj = EmbedAction(
                nclasses,
                self.latent_dim,
                guidance_scale=guidance_scale,
                guidance_uncodp=guidance_uncondp,
            )
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # === Position Encodings ===
        # Add positional information so the transformer knows sequence order
        if self.pe_type == "actor":
            # ACTOR-style: sinusoidal with dropout (adds to input)
            self.query_pos = PositionalEncoding(self.latent_dim, dropout)
            self.mem_pos = PositionalEncoding(self.latent_dim, dropout)
        elif self.pe_type == "mld":
            # MLD-style: learned or sinusoidal (returned separately, added in forward)
            self.query_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding
            )
            self.mem_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding
            )
        else:
            raise ValueError("Not Support PE type")

        # === Transformer Architecture ===
        # The denoiser uses transformer layers to process latent sequences
        if self.arch == "trans_enc":
            # Encoder-only architecture (default for MLD, faster than decoder)
            # Concatenates conditioning and noisy latent, processes together
            if self.ablation_skip_connection:
                # U-Net style with skip connections (default, better gradient flow)
                # Skip connections help preserve fine-grained details during denoising
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim,
                    num_heads,
                    ff_size,
                    dropout,
                    activation,
                    normalize_before,
                )
                encoder_norm = nn.LayerNorm(self.latent_dim)
                self.encoder = SkipTransformerEncoder(
                    encoder_layer, num_layers, encoder_norm
                )
            else:
                # Standard transformer encoder (no skip connections)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=num_layers
                )

        elif self.arch == "trans_dec":
            # Decoder architecture with cross-attention (more expressive but slower)
            # Uses cross-attention to attend to conditioning while processing latent
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_layers,
                decoder_norm,
                return_intermediate=return_intermediate_dec,
            )
        else:
            raise ValueError(f"Not supported architecture {self.arch}!")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        lengths: list[int] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Predict noise added to the latent sample at given timestep.

        This implements ε_θ(z_t, t, c) from the MLD paper, where:
        - z_t is the noisy latent at timestep t
        - t is the diffusion timestep
        - c is the conditioning (text or action)

        Args:
            sample: Noisy latent motion (batch_size, latent_size, latent_dim)
            timestep: Current diffusion timestep (scalar or batch_size,)
            encoder_hidden_states: Conditioning embeddings (batch_size, seq_len, dim)
            lengths: Sequence lengths for variable-length sequences

        Returns:
            Predicted noise (batch_size, latent_size, latent_dim)
        """
        # Transpose to (seq_len, batch_size, latent_dim) - transformer expects seq-first
        sample = sample.permute(1, 0, 2)

        # Create padding mask for variable-length sequences (if needed)
        if lengths not in [None, []]:
            mask = lengths_to_mask(lengths, sample.device)

        # === Step 1: Encode Timestep ===
        # Convert timestep to sinusoidal embedding, then project to latent_dim
        timesteps = timestep.expand(sample.shape[1]).clone()  # Broadcast to batch
        time_emb = self.time_proj(timesteps)  # Sinusoidal encoding
        time_emb = time_emb.to(dtype=sample.dtype)
        time_emb = self.time_embedding(time_emb).unsqueeze(0)  # (1, bs, latent_dim)

        # === Step 2: Prepare and Fuse Conditioning ===
        # Combine timestep and text/action embeddings for conditioning
        if self.condition in ["text", "text_uncond"]:
            # Text conditioning: use CLIP text embeddings
            encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)  # seq-first
            text_emb = encoder_hidden_states

            # Project text embeddings to latent dimension if needed
            if self.text_encoded_dim != self.latent_dim:
                text_emb_latent = self.emb_proj(text_emb)
            else:
                text_emb_latent = text_emb

            # Combine time and text embeddings
            if self.abl_plus:
                # Ablation: element-wise addition
                emb_latent = time_emb + text_emb_latent
            else:
                # Default: concatenate along sequence dimension
                # Result: [time_token, text_tokens...] to be processed with latent
                emb_latent = torch.cat((time_emb, text_emb_latent), 0)

        elif self.condition == "action":
            # Action conditioning: use learnable action embeddings
            action_emb = self.emb_proj(encoder_hidden_states)  # (1, bs, latent_dim)
            if self.abl_plus:
                emb_latent = action_emb + time_emb
            else:
                emb_latent = torch.cat((time_emb, action_emb), 0)
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # === Step 3: Transformer Processing ===
        # Process noisy latent with conditioning through transformer layers
        if self.arch == "trans_enc":
            # Encoder architecture: concatenate latent and conditioning, process jointly
            if self.diffusion_only:
                # Diffusion-only mode: project raw features first (not used in MLD)
                sample = self.pose_embd(sample)
                xseq = torch.cat((emb_latent, sample), axis=0)  # [cond..., latent...]
            else:
                # MLD mode: work directly with VAE latents
                # Concatenate: [latent_tokens..., time_token, text_tokens...]
                xseq = torch.cat((sample, emb_latent), axis=0)

            # Add positional encoding (either adds in-place or returns separately)
            xseq = self.query_pos(xseq)

            # Process through transformer encoder (with skip connections if enabled)
            tokens = self.encoder(xseq)

            # Extract the denoised latent tokens (first part of sequence)
            if self.diffusion_only:
                sample = tokens[emb_latent.shape[0] :]  # Skip conditioning tokens
                sample = self.pose_proj(sample)  # Project back to feature space
                sample[~mask.T] = 0  # Zero out padding
            else:
                sample = tokens[: sample.shape[0]]  # Extract latent portion

        elif self.arch == "trans_dec":
            # Decoder architecture: use cross-attention between latent and conditioning
            if self.diffusion_only:
                sample = self.pose_embd(sample)

            # Add position encodings to query (latent) and memory (conditioning)
            sample = self.query_pos(sample)
            emb_latent = self.mem_pos(emb_latent)

            # Cross-attend: latent queries attend to conditioning memory
            sample = self.decoder(tgt=sample, memory=emb_latent).squeeze(0)

            if self.diffusion_only:
                sample = self.pose_proj(sample)
                sample[~mask.T] = 0
        else:
            raise TypeError(f"{self.arch} is not supported")

        # === Step 4: Prepare Output ===
        # Transpose back to batch-first: (batch_size, latent_size, latent_dim)
        sample = sample.permute(1, 0, 2)

        return (sample,)  # Return as tuple for compatibility


class EmbedAction(nn.Module):
    """
    Learnable action embeddings with classifier-free guidance.

    Maps discrete action class labels to continuous embeddings. Supports
    classifier-free guidance by randomly masking embeddings during training,
    allowing the model to generate both conditionally and unconditionally.

    Classifier-free guidance (Ho & Salimans 2022) interpolates between conditional
    and unconditional predictions: ε = ε_uncond + w * (ε_cond - ε_uncond)
    """

    def __init__(
        self,
        num_actions: int,
        latent_dim: int,
        guidance_scale: float = 7.5,
        guidance_uncodp: float = 0.1,  # Unconditional probability during training
        force_mask: bool = False,
    ) -> None:
        super().__init__()
        self.nclasses = num_actions
        self.guidance_scale = guidance_scale
        # Learnable embedding table: one vector per action class
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))
        self.guidance_uncodp = guidance_uncodp
        self.force_mask = force_mask
        self._reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Action indices (batch_size, 1)

        Returns:
            Action embeddings (1, batch_size, latent_dim)
        """
        # Look up action embedding from learned table
        idx = input[:, 0].to(torch.long)
        output = self.action_embedding[idx]

        # Apply classifier-free guidance masking during inference
        if not self.training and self.guidance_scale > 1.0:
            # Split into unconditional and conditional batches
            uncond, output = output.chunk(2)
            uncond_out = self.mask_cond(uncond, force=True)  # Force zero for uncond
            out = self.mask_cond(output)
            output = torch.cat((uncond_out, out))

        # Apply conditional masking
        output = self.mask_cond(output)
        return output.unsqueeze(0)  # Add sequence dimension

    def mask_cond(self, output: torch.Tensor, force: bool = False) -> torch.Tensor:
        """
        Apply masking for classifier-free guidance.

        During training, randomly zero out embeddings with probability guidance_uncodp.
        This teaches the model to generate both with and without action conditioning.
        """
        bs, d = output.shape
        if self.force_mask or force:
            # Force unconditional (zero embedding)
            return torch.zeros_like(output)
        elif self.training and self.guidance_uncodp > 0.0:
            # Randomly mask embeddings during training
            mask = torch.bernoulli(
                torch.ones(bs, device=output.device) * self.guidance_uncodp
            ).view(bs, 1)
            return output * (1.0 - mask)
        else:
            return output

    def _reset_parameters(self) -> None:
        """Initialize embedding table with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
