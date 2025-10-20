---
title: Motion Latent Diffusion Standalone Demo
emoji: 📊
colorFrom: purple
colorTo: indigo
sdk: gradio
python_version: 3.11
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# Demo

Command-line and web interfaces for motion-latent-diffusion-standalone.

## Installation

```bash
cd demo
pip install -e .
```

## Command Line

```bash
# Generate motion
python cli.py --text "a person walks forward" --length 100

# Options
python cli.py --text "jumping" --length 120 --output ./outputs/ --no-video
```

Outputs:

- `*.pt` - Motion tensor (frames, 22, 3)
- `*.latent.pt` - Latent representation
- `*.mp4` - Visualization video
- `*.txt` - Text prompt

## Web Interface

```bash
python app.py
```

Opens at `http://localhost:7860`

## Visualization

```bash
# Create video from saved motion
python visualize.py motion.pt --output video.mp4 --fps 20
```

## Python API

```python
from motion_latent_diffusion_standalone import MotionLatentDiffusionModel
from visualize import create_video_from_joints

model = MotionLatentDiffusionModel(
    vae_repo_id="blanchon/motion-latent-diffusion-standalone-vae",
    denoiser_repo_id="blanchon/motion-latent-diffusion-standalone-denoiser"
)

joints = model.generate("a person walks", length=100)  # (100, 22, 3)
create_video_from_joints(joints, "output.mp4", fps=20)
```
