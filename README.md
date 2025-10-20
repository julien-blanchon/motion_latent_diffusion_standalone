# Motion Latent Diffusion (Standalone)

A minimal, self-contained implementation of [Motion Latent Diffusion](https://github.com/ChenFengYe/motion-latent-diffusion) for text-to-motion generation.

Original paper: [Executing your Commands via Motion Diffusion in Latent Space](https://arxiv.org/abs/2212.04048) (Chen et al., CVPR 2023)

## Installation

```bash
pip install motion-latent-diffusion-standalone
```

## Usage

```python
from motion_latent_diffusion_standalone import MotionLatentDiffusionModel

# Load model (downloads checkpoints on first run)
model = MotionLatentDiffusionModel(
    vae_repo_id="blanchon/motion-latent-diffusion-standalone-vae",
    denoiser_repo_id="blanchon/motion-latent-diffusion-standalone-denoiser"
)

# Generate motion from text
joints = model.generate("a person walks forward", length=100)
# Returns: (100, 22, 3) tensor - 100 frames, 22 joints, XYZ coordinates
```

## Model Checkpoints

Pre-trained models on Hugging Face:

- [Denoiser](https://huggingface.co/blanchon/motion-latent-diffusion-standalone-denoiser)
- [VAE](https://huggingface.co/blanchon/motion-latent-diffusion-standalone-vae)

## Demo

Try the [interactive demo](https://huggingface.co/spaces/blanchon/motion-latent-diffusion-standalone) on Hugging Face Spaces.

Or run locally:

```bash
cd demo
pip install -e .
python app.py  # Web interface at localhost:7860
python cli.py --text "a person jumps"  # Command line
```

## Architecture

The model consists of three components:

1. **Text Encoder**: CLIP (frozen) converts text to embeddings
2. **Denoiser**: Transformer-based diffusion model in latent space
3. **VAE**: Encodes/decodes motion sequences to/from latent space

Motion representation: 22-joint skeleton (HumanML3D format) at 20 FPS.

Latent space: 1×256 dimension per sequence (enables efficient diffusion).

## Citation

```bibtex
@inproceedings{chen2023executing,
  title={Executing your Commands via Motion Diffusion in Latent Space},
  author={Chen, Xin and Jiang, Biao and Liu, Wen and Huang, Zilong and Fu, Bin and Chen, Tao and Yu, Gang},
  booktitle={CVPR},
  year={2023}
}
```

## License

MIT
