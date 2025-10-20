"""
Utility Functions for Model Operations

Helper functions used across the model architecture, particularly for
handling variable-length sequences in transformers.
"""

from collections.abc import Sequence
import torch
from torch import Tensor


def lengths_to_mask(
    lengths: Sequence[int], device: torch.device, max_len: int | None = None
) -> Tensor:
    """
    Convert sequence lengths to a boolean padding mask.

    Transformers need to know which positions contain real data vs padding.
    This function creates a mask where True = valid data, False = padding.

    Example:
        lengths = [3, 5, 2]
        mask = [[True, True, True, False, False],
                [True, True, True, True, True],
                [True, True, False, False, False]]

    Args:
        lengths: List of actual sequence lengths (e.g., [3, 5, 2])
        device: Device to create tensor on
        max_len: Maximum sequence length (if None, uses max of lengths)

    Returns:
        Boolean mask (batch_size, max_len) where True = valid, False = padding
    """
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)

    # Create position indices: [0, 1, 2, ..., max_len-1]
    # Broadcast to (batch_size, max_len) and compare with lengths
    # Result: position < length → True (valid), position >= length → False (padding)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask
