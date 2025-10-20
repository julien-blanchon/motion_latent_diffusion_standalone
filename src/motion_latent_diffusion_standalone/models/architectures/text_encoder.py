"""
MLD Text Encoder using CLIP

Encodes text prompts into embeddings for conditioning the diffusion model.
Supports CLIP and BERT models from Hugging Face transformers.
"""

from typing import Literal
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """
    Text encoder using CLIP or BERT models from Hugging Face transformers.

    Encodes text prompts into embeddings for conditioning the diffusion model.

    Args:
        modelpath: Path to pretrained model on Hugging Face Hub
        finetune: Whether to finetune the model (default: False)
        last_hidden_state: For CLIP, use last hidden state instead of pooled output
        latent_dim: Latent dimension configuration [size, dim]
    """

    def __init__(
        self,
        modelpath: str,
        finetune: bool = False,
        last_hidden_state: bool = False,
        latent_dim: list[int] = [1, 256],
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        # Load CLIP tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.text_model = AutoModel.from_pretrained(modelpath)

        # Freeze model if not finetuning
        if not finetune:
            self.text_model.train(False)
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Configure model based on architecture
        self.max_length = self.tokenizer.model_max_length

        if "clip" in modelpath.lower():
            self.text_encoded_dim: int = self.text_model.config.text_config.hidden_size
            self.model_type: Literal["clip", "clip_hidden", "bert"] = (
                "clip_hidden" if last_hidden_state else "clip"
            )
        elif "bert" in modelpath.lower():
            self.model_type = "bert"
            self.text_encoded_dim = self.text_model.config.hidden_size
        else:
            raise ValueError(f"Model {modelpath} not supported")

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Encode text prompts into embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            text_embeddings: Tensor of shape (batch_size, seq_len, text_encoded_dim)
                            For CLIP pooled: (batch_size, 1, text_encoded_dim)
                            For CLIP hidden/BERT: (batch_size, seq_length, text_encoded_dim)
        """
        # Tokenize text
        if self.model_type in ["clip", "clip_hidden"]:
            text_inputs = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            # Clip to max length if needed
            if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

        elif self.model_type == "bert":
            text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
            text_input_ids = text_inputs.input_ids
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented")

        # Move to model device
        device = next(self.text_model.parameters()).device
        text_input_ids = text_input_ids.to(device)

        # Encode text
        if self.model_type == "clip":
            # Use pooled output (batch_size, text_encoded_dim)
            text_embeddings = self.text_model.get_text_features(text_input_ids)
            # Add sequence dimension (batch_size, 1, text_encoded_dim)
            text_embeddings = text_embeddings.unsqueeze(1)

        elif self.model_type == "clip_hidden":
            # Use last hidden state (batch_size, seq_length, text_encoded_dim)
            text_embeddings = self.text_model.text_model(
                text_input_ids
            ).last_hidden_state

        elif self.model_type == "bert":
            # Use last hidden state (batch_size, seq_length, text_encoded_dim)
            text_embeddings = self.text_model(text_input_ids).last_hidden_state

        else:
            raise NotImplementedError(f"Model {self.model_type} not implemented")

        return text_embeddings
