"""
Transformer Layers with U-Net Style Skip Connections

Adapted from DETR (Facebook Research) with skip connections inspired by U-Net.
These transformers are used in both the VAE and denoiser architectures.

Key features:
- SkipTransformerEncoder/Decoder: U-Net style with skip connections between layers
- TransformerEncoder/Decoder: Standard transformer without skips
- Skip connections preserve fine-grained information during deep processing
- Particularly useful for reconstruction tasks (VAE) and denoising

Skip connection architecture:
- Input blocks (encoder path): L//2 layers, save activations
- Middle block: 1 layer (bottleneck)
- Output blocks (decoder path): L//2 layers, receive skip connections
- Total layers must be odd (e.g., 9 = 4 input + 1 middle + 4 output)
"""

import copy
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _get_clone(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor, ...], Tensor]:
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class SkipTransformerEncoder(nn.Module):
    """
    Transformer encoder with U-Net style skip connections.

    Adds skip connections between mirrored layers to preserve fine-grained
    information through deep networks. This is particularly beneficial for
    reconstruction (VAE) and denoising tasks where details matter.

    Architecture for num_layers=9:
    Input path:     [L1] -> [L2] -> [L3] -> [L4] -> [Middle]
                     |       |       |       |
    Output path:    [L5] <- [L6] <- [L7] <- [L8] <- [L9]

    Skip connections concatenate features: [output, input] -> Linear -> continue
    """

    def __init__(
        self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.d_model = encoder_layer.d_model
        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1, "num_layers must be odd for U-Net structure"

        # Build U-Net structure: input path -> bottleneck -> output path
        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(encoder_layer, num_block)  # Encoder path
        self.middle_block = _get_clone(encoder_layer)  # Bottleneck
        self.output_blocks = _get_clones(encoder_layer, num_block)  # Decoder path

        # Linear layers to merge concatenated skip connections
        # [current_features, skip_features] -> d_model
        self.linear_blocks = _get_clones(
            nn.Linear(2 * self.d_model, self.d_model), num_block
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            src: Input sequence (seq_len, batch_size, d_model)
            mask: Attention mask
            src_key_padding_mask: Padding mask (batch_size, seq_len)
            pos: Position encoding to add to src

        Returns:
            Output sequence (seq_len, batch_size, d_model)
        """
        x = src
        xs = []  # Stack to store skip connection features

        # === Encoder Path (Input Blocks) ===
        # Process through first half of layers, saving activations
        for module in self.input_blocks:
            x = module(
                x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )
            xs.append(x)  # Save for skip connection

        # === Bottleneck (Middle Block) ===
        # Process through the deepest layer
        x = self.middle_block(
            x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
        )

        # === Decoder Path (Output Blocks with Skip Connections) ===
        # Process through second half, merging with saved features
        for module, linear in zip(self.output_blocks, self.linear_blocks):
            # Retrieve skip connection from encoder path (LIFO order)
            skip = xs.pop()

            # Concatenate current features with skip connection
            x = torch.cat([x, skip], dim=-1)  # (seq, batch, 2*d_model)

            # Project back to d_model
            x = linear(x)  # (seq, batch, d_model)

            # Process through transformer layer
            x = module(
                x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )

        # Final normalization
        if self.norm is not None:
            x = self.norm(x)
        return x


class SkipTransformerDecoder(nn.Module):
    """Transformer decoder with U-Net style skip connections"""

    def __init__(
        self, decoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.d_model = decoder_layer.d_model
        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1, "num_layers must be odd for skip connections"

        # Build U-Net structure for decoder
        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(decoder_layer, num_block)
        self.middle_block = _get_clone(decoder_layer)
        self.output_blocks = _get_clones(decoder_layer, num_block)
        # Linear layers for concatenation in skip connections
        self.linear_blocks = _get_clones(
            nn.Linear(2 * self.d_model, self.d_model), num_block
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        x = tgt
        xs = []

        # Decoder path (downward): apply input blocks and save features
        for module in self.input_blocks:
            x = module(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            xs.append(x)

        # Middle block: bottleneck of U-Net
        x = self.middle_block(
            x,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
        )

        # Decoder path (upward) with skip connections
        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )

        if self.norm is not None:
            x = self.norm(x)

        return x


class TransformerEncoder(nn.Module):
    """Standard transformer encoder"""

    def __init__(
        self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """Standard transformer decoder"""

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: nn.Module | None = None,
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feedforward"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        # Self-attention with positional embeddings
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feedforward network with residual connection
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        # Pre-norm variant: normalize before operations
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention, cross-attention and feedforward"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        # Self-attention on target
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Cross-attention between target and memory
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        # Self-attention on target
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention between target and memory
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        # Self-attention (pre-norm)
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        # Cross-attention (pre-norm)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        # Feedforward (pre-norm)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )
