"""
MLD Demo CLI - Generate human motion from text using the standalone MLD package.
"""

import argparse
from pathlib import Path
from datetime import datetime
import torch
from textwrap import dedent
from tqdm import tqdm

from motion_latent_diffusion_standalone import MotionLatentDiffusionModel
from visualize import create_video_from_joints


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate human motion from text using MLD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
            Examples:
            # Basic usage
            python cli.py --text "a person walks forward slowly"
            
            # Custom length
            python cli.py --text "jumping jacks" --length 120
            
            # Save to specific directory
            python cli.py --text "dancing" --output ./motions/
            
            # Skip video generation (faster)
            python cli.py --text "running" --no-video
        """),
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text description of the motion to generate",
    )

    parser.add_argument(
        "--length",
        type=int,
        default=100,
        help="Motion length in frames (default: 100, range: 16-196 for 20fps)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Output directory for generated files (default: ./outputs)",
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video generation, only save .pt file",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda if available, else cpu)",
    )

    return parser.parse_args()


def generate_filename(text: str) -> str:
    """Generate a filename from text and timestamp"""
    # Clean text for filename: remove special characters
    text_clean = "".join(c if c.isalnum() or c.isspace() else "" for c in text)
    text_clean = "_".join(text_clean.split()[:5])  # First 5 words
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{text_clean}_{timestamp}"


def main() -> None:
    """Main entry point for CLI"""
    args = parse_args()

    # Validate motion length
    if args.length < 16 or args.length > 196:
        print(f"Warning: Length {args.length} is outside recommended range (16-196)")
        print("Proceeding anyway, but results may be suboptimal.")

    # Setup output paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filenames
    base_name = generate_filename(args.text)
    pt_path = output_dir / f"{base_name}.pt"
    mp4_path = output_dir / f"{base_name}.mp4"
    txt_path = output_dir / f"{base_name}.txt"

    print("=" * 70)
    print("MLD Text-to-Motion Generator")
    print("=" * 70)
    print(f"Text prompt: {args.text}")
    print(f"Motion length: {args.length} frames ({args.length / 20:.1f}s at 20fps)")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Device: {args.device}")
    print("=" * 70)

    # [1/4] Load model from HuggingFace Hub
    print("\n[1/4] Loading model from HuggingFace Hub...")
    print("This may take a minute on first run (downloads ~105MB)...")
    model = MotionLatentDiffusionModel(
        vae_repo_id="blanchon/motion-latent-diffusion-standalone-vae",
        denoiser_repo_id="blanchon/motion-latent-diffusion-standalone-denoiser",
        text_encoder_repo_id="openai/clip-vit-large-patch14",
    ).to(args.device)

    # [2/4] Generate motion
    print("\n[2/4] Generating motion...")
    print(f"Running diffusion sampling ({model.num_inference_timesteps} steps)...")

    with tqdm(total=args.length, desc="Generating motion") as pbar:

        def callback_on_step_end(i: int, latents: torch.Tensor):
            pbar.update(i)

        # Generate motion (returns PyTorch tensor)
        joints, latent = model.generate(
            args.text,
            args.length,
            return_latent=True,
            callback_on_step_end=callback_on_step_end,
        )

    print(f"✓ Generated motion: {joints.shape}")
    print(
        f"  Shape: ({joints.shape[0]} frames, {joints.shape[1]} joints, {joints.shape[2]} coords)"
    )

    # [3/4] Save motion file as PyTorch tensor
    print("\n[3/4] Saving files...")
    torch.save(joints, pt_path)
    print(f"✓ Saved motion: {pt_path}")

    # Save latent representation
    latent_path = output_dir / f"{base_name}.latent.pt"
    torch.save(latent, latent_path)
    print(f"✓ Saved latent: {latent_path}")

    # Save text prompt for reference
    with open(txt_path, "w") as f:
        f.write(args.text)
    print(f"✓ Saved prompt: {txt_path}")

    # [4/4] Generate video if requested
    if not args.no_video:
        print("\n[4/4] Generating video visualization...")
        video_path = create_video_from_joints(joints, str(mp4_path), fps=20)
        print(f"✓ Generated video: {video_path}")
    else:
        print("\n[4/4] Skipping video generation (--no-video flag)")

    # Print summary
    print("\n" + "=" * 70)
    print("✓ Generation complete!")
    print("=" * 70)
    print("Output files:")
    print(f"  Motion data: {pt_path}")
    print(f"  Latent repr: {latent_path}")
    print(f"  Text prompt: {txt_path}")
    if not args.no_video:
        print(f"  Video:       {mp4_path}")
    print("\nTo visualize the motion later:")
    print(f"  python visualize.py {pt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
