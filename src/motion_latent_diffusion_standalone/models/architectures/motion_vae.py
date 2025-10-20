"""
Motion Variational Autoencoder (VAE)

Compresses motion sequences into a low-dimensional latent space for efficient
diffusion modeling. As described in MLD paper Section 3.1, the VAE learns a
representative latent distribution z ~ q(z|x) that captures motion semantics
while reducing dimensionality.

Key benefits (from paper):
- Reduces computational cost by 2 orders of magnitude vs raw motion diffusion
- Provides better information density in latent space
- Can be pretrained on large-scale unlabeled motion datasets (AMASS)
- Achieves state-of-the-art reconstruction quality

Architecture (Table 13):
- Transformer encoder-decoder with 9 layers, 4 heads, 256 hidden dim
- U-Net style skip connections for better reconstruction
- Latent space: [1, 256] (1 token of 256 dimensions)
"""

from typing import Literal
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.distribution import Distribution
from huggingface_hub import PyTorchModelHubMixin

from ..operators import PositionalEncoding
from ..operators.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerEncoderLayer,
)
from ..operators.position_encoding import build_position_encoding
from ..utils import lengths_to_mask
from ...transforms import recover_from_ric


class MotionVAE(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/julien-blanchon/motion_latent_diffusion_standalone",
):
    """
    Motion Variational Autoencoder with Transformer Architecture.

    Learns to encode motion sequences x^{1:L} into a low-dimensional latent
    distribution z ~ N(μ, σ²), then decode back to motion. The latent space
    provides a compressed, semantically meaningful representation for diffusion.

    The VAE is trained with:
    - Reconstruction loss: ||x - D(E(x))||²
    - KL regularization: KL(q(z|x) || p(z)) where p(z) = N(0, I)

    Features:
    - U-Net style skip connections between encoder and decoder layers
    - Learnable motion tokens for latent distribution parameterization
    - Supports both SMPL and joint-based motion representations
    """

    def __init__(
        self,
        nfeats: int,
        latent_dim: list[int] = [1, 256],
        ff_size: int = 1024,
        num_layers: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1,
        arch: Literal["all_encoder", "encoder_decoder"] = "all_encoder",
        normalize_before: bool = False,
        activation: Literal["gelu", "relu"] = "gelu",
        position_embedding: Literal["sine", "learned", "v2", "v3"] = "learned",
        mlp_dist: bool = False,
        pe_type: Literal["actor", "mld"] = "mld",
        njoints: int = 22,  # Number of joints for motion data
        **kwargs,
    ) -> None:
        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        self.nfeats = nfeats
        self.njoints = njoints
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = mlp_dist
        self.pe_type = pe_type

        # Register mean and std as buffers (will be saved with the model)
        # Initialize with None to force proper initialization
        # These MUST be set before use via set_normalization()
        self.mean: nn.Tensor
        self.std: nn.Tensor
        self.register_buffer("mean", torch.zeros(nfeats))
        self.register_buffer("std", torch.ones(nfeats))

        # Position encoding setup (actor or mld style)
        if self.pe_type == "actor":
            self.query_pos_encoder = PositionalEncoding(self.latent_dim, dropout)
            self.query_pos_decoder = PositionalEncoding(self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.query_pos_encoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding
            )
            self.query_pos_decoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding
            )
        else:
            raise ValueError("Not Support PE type")

        # Encoder with optional skip connections
        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)

        # Decoder architecture (all_encoder or encoder_decoder)
        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerEncoder(
                encoder_layer, num_layers, decoder_norm
            )
        elif self.arch == "encoder_decoder":
            from ..operators.cross_attention import TransformerDecoderLayer

            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(
                decoder_layer, num_layers, decoder_norm
            )
        else:
            raise ValueError("Not support architecture!")

        # Latent distribution parameterization
        if self.mlp_dist:
            # Learnable motion tokens for mean/logvar prediction
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim)
            )
            self.dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
        else:
            # Split tokens for separate mean and logvar
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim)
            )

        # Input/output projections
        self.skel_embedding = nn.Linear(input_feats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, output_feats)

    def forward(
        self, features: Tensor, lengths: list[int] | None = None
    ) -> tuple[Tensor, Tensor, Distribution]:
        """Forward pass (not used in inference)"""
        print("Should Not enter here")
        z, dist = self.encode(features, lengths)
        feats_rst = self.decode(z, lengths)
        return feats_rst, z, dist

    def encode(
        self, features: Tensor, lengths: list[int] | None = None
    ) -> tuple[Tensor, Distribution]:
        """
        Encode motion features x into latent distribution q(z|x).

        The encoder processes motion sequences through a transformer and outputs
        parameters (μ, σ) of a Gaussian latent distribution. A latent sample z
        is drawn using the reparameterization trick: z = μ + σ * ε, ε ~ N(0,1).

        Args:
            features: Motion features (batch_size, nframes, nfeats)
            lengths: Sequence lengths for variable-length motions

        Returns:
            latent: Sampled latent vector z (latent_size, batch_size, latent_dim)
            dist: Latent distribution q(z|x) ~ N(μ, σ²)
        """
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device
        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)  # True for valid positions

        # Project motion features to latent dimension
        x = self.skel_embedding(features)

        # Transpose to sequence-first for transformer: (nframes, bs, latent_dim)
        x = x.permute(1, 0, 2)

        # Learnable motion tokens that will absorb the motion information
        # These tokens are repeated across the batch
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # Create augmented mask: [motion_tokens (always valid), motion_frames (use mask)]
        dist_masks = torch.ones((bs, dist.shape[0]), dtype=bool, device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # Concatenate tokens and motion: [motion_tokens..., motion_frames...]
        xseq = torch.cat((dist, x), 0)

        # Encode through transformer with position encoding
        if self.pe_type in ["actor", "mld"]:
            xseq = self.query_pos_encoder(xseq)
            # Process through encoder, then extract just the motion tokens
            dist = self.encoder(xseq, src_key_padding_mask=~aug_mask)[: dist.shape[0]]

        # Extract distribution parameters from motion tokens
        if self.mlp_dist:
            # Project tokens through MLP to get [μ, log σ²]
            tokens_dist = self.dist_layer(dist)
            mu = tokens_dist[:, :, : self.latent_dim]
            logvar = tokens_dist[:, :, self.latent_dim :]
        else:
            # Split tokens directly: first half = μ, second half = log σ²
            mu = dist[0 : self.latent_size, ...]
            logvar = dist[self.latent_size :, ...]

        # Sample from latent distribution using reparameterization trick
        # z = μ + σ * ε, where ε ~ N(0, 1)
        std = logvar.exp().pow(0.5)  # σ = exp(log σ² / 2)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()  # Reparameterized sampling (differentiable)

        return latent, dist

    def decode(self, z: Tensor, lengths: list[int]) -> Tensor:
        """
        Decode latent vector z back to motion features x̂.

        The decoder expands the compressed latent representation back into a full
        motion sequence. It uses either encoder-only (concatenation) or decoder
        (cross-attention) architecture with skip connections from the encoder.

        Args:
            z: Latent vectors (latent_size, batch_size, latent_dim)
            lengths: Target sequence lengths for each batch element

        Returns:
            feats: Reconstructed motion features (batch_size, nframes, nfeats)
        """
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        # Initialize learnable query tokens for each output frame
        # These will be filled with motion information through attention
        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)

        # Decode through transformer (two architectural options)
        if self.arch == "all_encoder":
            # Encoder-only architecture: concatenate latent and queries, process together
            # Structure: [latent_tokens..., query_tokens...]
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, self.latent_size), dtype=bool, device=z.device)
            augmask = torch.cat((z_mask, mask), axis=1)

            if self.pe_type in ["actor", "mld"]:
                xseq = self.query_pos_decoder(xseq)  # Add position encoding
                # Process through decoder, extract query portion
                output = self.decoder(xseq, src_key_padding_mask=~augmask)[z.shape[0] :]

        elif self.arch == "encoder_decoder":
            # True decoder: cross-attention from queries to latent memory
            # Queries attend to latent, extracting relevant motion information
            if self.pe_type in ["actor", "mld"]:
                queries = self.query_pos_decoder(queries)
                output = self.decoder(
                    tgt=queries,  # What we want to fill
                    memory=z,  # Where to get information from
                    tgt_key_padding_mask=~mask,
                ).squeeze(0)

        # Project from latent_dim back to motion feature dimension
        output = self.final_layer(output)

        # Zero out padded positions (ensures no information leakage)
        output[~mask.T] = 0

        # Convert back to batch-first: (batch_size, nframes, nfeats)
        feats = output.permute(1, 0, 2)

        return feats

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """
        Set the normalization parameters for motion data.

        Args:
            mean: Mean values for denormalization
            std: Standard deviation values for denormalization
        """
        self.mean.copy_(mean)
        self.std.copy_(std)

    def feats2joints(self, features: Tensor) -> Tensor:
        """
        Convert normalized motion features to 3D joint positions.

        This is a convenience method that combines:
        1. Denormalization: features * std + mean
        2. RIC to joints conversion: recover root position/rotation and local joints

        The motion features use RIC (Rotation-Invariant Coordinates) representation
        from HumanML3D, which encodes motion relative to the root. This function
        recovers the absolute 3D joint positions in world space.

        Args:
            features: Normalized motion features (batch, seq_len, nfeats)

        Returns:
            joints: 3D joint positions in world space (batch, seq_len, njoints, 3)
        """

        # Denormalize: convert from standardized to original scale
        mean = self.mean.to(features.device)
        std = self.std.to(features.device)
        features = features * std + mean

        # Convert from RIC representation to 3D joint coordinates
        # This recovers root rotation/position and applies to local joint positions
        joints = recover_from_ric(features, self.njoints)

        return joints
