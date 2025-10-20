"""
Motion Latent Diffusion (MLD) - Main Model

Combines VAE and diffusion model for efficient text-to-motion generation.
As described in the MLD paper, this architecture achieves 2 orders of magnitude
speedup over raw motion diffusion while maintaining quality.

Pipeline (Section 3):
1. VAE Encoder: x → z (compress motion to latent)
2. Diffusion Forward: z → z_t (add noise during training)
3. Denoiser: ε_θ(z_t, t, text) (predict noise conditioned on text)
4. DDIM Sampling: z_t → z (iteratively denoise during inference)
5. VAE Decoder: z → x̂ (reconstruct motion)

Key advantages:
- 100x faster inference than MDM (raw motion diffusion)
- Can train VAE on large unlabeled datasets (AMASS)
- Better sample quality through learned latent space
- Supports classifier-free guidance for controllable generation
"""

from typing import Callable
import torch
from diffusers import DDIMScheduler
from torch import nn
from huggingface_hub import PyTorchModelHubMixin

from .models import TextEncoder, MotionVAE, Denoiser


class MotionLatentDiffusionModel(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/julien-blanchon/motion_latent_diffusion_standalone",
):
    """
    Motion Latent Diffusion Model for Text-to-Motion Generation.

    This is the main interface that combines the VAE, denoiser, and text encoder
    into a complete text-to-motion generation pipeline. The model operates in
    a learned latent space for efficiency.

    Loading options:

    1. From HuggingFace Hub (recommended):
       ```python
       model = MotionLatentDiffusionModel.from_pretrained(
           "blanchon/motion-latent-diffusion-standalone"
       )
       ```

    2. By providing component repo_ids:
       ```python
       model = MotionLatentDiffusionModel(
           vae_repo_id="blanchon/motion-latent-diffusion-standalone-vae",
           denoiser_repo_id="blanchon/motion-latent-diffusion-standalone-denoiser"
       )
       ```

    3. By providing pre-loaded components:
       ```python
       model = MotionLatentDiffusionModel(
           vae=vae_module,
           denoiser=denoiser_module
       )
       ```
    """

    def __init__(
        self,
        # Component repo IDs (for loading from Hub) - JSON-serializable
        vae_repo_id: str | None = None,
        denoiser_repo_id: str | None = None,
        text_encoder_repo_id: str = "openai/clip-vit-large-patch14",
        # OR pre-loaded components (not serialized)
        vae: nn.Module | None = None,
        denoiser: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
        # Model parameters - JSON-serializable
        nfeats: int = 263,
        njoints: int = 22,
        latent_dim: list[int] = [1, 256],
        # Guidance parameters
        guidance_scale: float = 7.5,
        # Scheduler parameters
        num_train_timesteps: int = 1000,
        num_inference_timesteps: int = 50,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ) -> None:
        super().__init__()

        # Store config (JSON-serializable only for PyTorchModelHubMixin)
        self.vae_repo_id = vae_repo_id
        self.denoiser_repo_id = denoiser_repo_id
        self.text_encoder_repo_id = text_encoder_repo_id
        self.nfeats = nfeats
        self.njoints = njoints
        self.latent_dim = latent_dim
        self.guidance_scale = guidance_scale
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.do_classifier_free_guidance = guidance_scale > 1.0

        # Load or use provided components
        self._init_components(
            vae_repo_id,
            denoiser_repo_id,
            text_encoder_repo_id,
            vae,
            denoiser,
            text_encoder,
            latent_dim,
        )

        # Initialize scheduler
        self._init_scheduler(
            num_train_timesteps, beta_start, beta_end, num_inference_timesteps
        )

    def _init_components(
        self,
        vae_repo_id: str | None,
        denoiser_repo_id: str | None,
        text_encoder_repo_id: str,
        vae: nn.Module | None,
        denoiser: nn.Module | None,
        text_encoder: nn.Module | None,
        latent_dim: list[int],
    ) -> None:
        """Initialize model components from repos or provided modules"""

        # TextEncoder
        if text_encoder is not None:
            self.text_encoder = text_encoder
        else:
            print(f"Loading TextEncoder from {text_encoder_repo_id}...")
            self.text_encoder = TextEncoder(
                modelpath=text_encoder_repo_id,
                finetune=False,
                last_hidden_state=False,
                latent_dim=latent_dim,
            )
        self.text_encoder.eval()

        # MotionVAE
        if vae is not None:
            self.vae = vae
        elif vae_repo_id is not None:
            print(f"Loading MotionVAE from {vae_repo_id}...")
            self.vae = MotionVAE.from_pretrained(vae_repo_id)
        else:
            raise ValueError("Either vae_repo_id or vae must be provided")
        self.vae.eval()

        # Denoiser
        if denoiser is not None:
            self.denoiser = denoiser
        elif denoiser_repo_id is not None:
            print(f"Loading Denoiser from {denoiser_repo_id}...")
            self.denoiser = Denoiser.from_pretrained(denoiser_repo_id)
        else:
            raise ValueError("Either denoiser_repo_id or denoiser must be provided")
        self.denoiser.eval()

    def _init_scheduler(
        self,
        num_train_timesteps: int,
        beta_start: float,
        beta_end: float,
        num_inference_timesteps: int,
    ) -> None:
        """Initialize DDIM scheduler for inference"""
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

    @torch.no_grad()
    def generate(
        self,
        text: str,
        length: int,
        return_latent: bool = False,
        device: str | None = None,
        callback_on_step_end: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Generate motion from text description.

        This is the main inference method that runs the complete MLD pipeline:
        1. Encode text with CLIP
        2. Generate latent via DDIM sampling (denoising from noise)
        3. Decode latent to motion features via VAE decoder
        4. Convert features to 3D joint positions

        Uses classifier-free guidance (CFG) for better text alignment:
        ε = ε_uncond + w * (ε_text - ε_uncond), where w = guidance_scale

        Args:
            text: Natural language description (e.g., "a person walks forward")
            length: Motion duration in frames (typically 20 FPS, so 60 = 3 seconds)
            return_latent: If True, also return the generated latent z
            device: Device to run inference on (default: model's current device)
            callback_on_step_end: Optional callback(step_idx, latents) after each denoising step

        Returns:
            joints: 3D joint positions (length, 22, 3) - ready for visualization
            latent: (optional) Generated latent vector if return_latent=True
        """
        # Determine device
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        # Move models to device if needed
        self.to(device)

        # Prepare text embeddings for classifier-free guidance
        texts = [text]
        if self.do_classifier_free_guidance:
            # For CFG, we need both conditional (with text) and unconditional (empty string)
            uncond_tokens = [""] * len(texts)
            texts = uncond_tokens + texts  # [uncond, cond]

        # Encode text through CLIP (frozen encoder)
        text_emb = self.text_encoder(texts)

        # Generate latent via diffusion reverse process (DDIM sampling)
        lengths = [length]
        z = self._diffusion_reverse(text_emb, lengths, device, callback_on_step_end)

        # Decode latent to motion features through VAE decoder
        feats = self.vae.decode(z, lengths)

        # Convert normalized features to 3D joint positions
        # This handles denormalization and RIC → joints conversion
        joints = self.vae.feats2joints(feats.cpu())
        joints_tensor = joints[0]  # Extract from batch: (length, 22, 3)

        # Return joints and optionally the latent
        if return_latent:
            return joints_tensor, z.cpu()
        return joints_tensor

    def _diffusion_reverse(
        self,
        encoder_hidden_states: torch.Tensor,
        lengths: list[int],
        device: torch.device,
        callback_on_step_end: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """
        Run DDIM sampling to generate motion latent from noise.

        Implements the reverse diffusion process (Algorithm 2 in DDIM paper):
        Starting from pure noise z_T ~ N(0, I), iteratively denoise using the
        predicted noise ε_θ(z_t, t, text) to arrive at a clean latent z_0.

        DDIM allows fast sampling with fewer steps (50 instead of 1000) by
        using a deterministic sampling trajectory instead of the full Markov chain.

        Classifier-free guidance (if enabled) improves text alignment:
        ε = ε_uncond + w * (ε_text - ε_uncond)
        where w = guidance_scale (typically 7.5)

        Args:
            encoder_hidden_states: Text embeddings (batch_size, seq_len, dim)
                                  If CFG enabled: [uncond_emb, cond_emb]
            lengths: Motion sequence lengths
            device: Device for computation
            callback_on_step_end: Optional callback after each step

        Returns:
            latents: Generated clean latent z_0 (batch_size, latent_size, latent_dim)
        """
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2  # Split back into single batch

        # Initialize with pure Gaussian noise: z_T ~ N(0, I)
        latents = torch.randn(
            (bsz, self.latent_dim[0], self.latent_dim[-1]),
            device=device,
            dtype=torch.float,
        )

        # Scale by scheduler's initial noise level (for numerical stability)
        latents = latents * self.scheduler.init_noise_sigma

        # Set up DDIM sampling schedule (maps 1000 train steps to 50 inference steps)
        self.scheduler.set_timesteps(self.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device)  # [T, T-1, ..., 1]

        # === Denoising Loop ===
        # Iteratively predict and remove noise: z_t → z_{t-1} → ... → z_0
        for i, t in enumerate(timesteps):
            # For CFG, we need to run the model twice: once unconditional, once conditional
            latent_model_input = (
                torch.cat([latents] * 2)  # Duplicate: [uncond, cond]
                if self.do_classifier_free_guidance
                else latents
            )

            # Prepare lengths (duplicate for CFG)
            lengths_reverse = (
                torch.tensor(lengths * 2, device=device)
                if self.do_classifier_free_guidance
                else torch.tensor(lengths, device=device)
            )

            # Predict noise: ε_θ(z_t, t, text)
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )

            # Extract prediction (some models return tuples)
            if isinstance(noise_pred, tuple):
                noise_pred = noise_pred[0]

            # Apply classifier-free guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # Guidance formula: move away from uncond towards text-cond
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # DDIM step: compute z_{t-1} from z_t and predicted noise
            # This uses the deterministic DDIM formula for fast sampling
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Optional callback (useful for visualization/debugging)
            if callback_on_step_end is not None:
                callback_on_step_end(i, latents)

        return latents  # Return clean latent z_0
