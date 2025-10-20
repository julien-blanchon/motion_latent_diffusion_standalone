"""
Simple 3D skeleton motion visualizer for HumanML3D motion data.
Usage: python visualize.py <motion.pt> [--output output.mp4] [--fps 20]
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path


# HumanML3D skeleton structure (22 joints)
# Kinematic chain based on HumanML3D dataset specification
# From mld/utils/joints.py and datasets/HumanML3D/paramUtil.py
SKELETON_CHAINS = [
    [0, 3, 6, 9, 12, 15],  # Body: root -> BP -> BT -> BLN -> BMN -> BUN (head)
    [9, 14, 17, 19, 21],  # Left arm: BLN -> LSI -> LS -> LE -> LW
    [9, 13, 16, 18, 20],  # Right arm: BLN -> RSI -> RS -> RE -> RW
    [0, 2, 5, 8, 11],  # Left leg: root -> LH -> LK -> LMrot -> LF
    [0, 1, 4, 7, 10],  # Right leg: root -> RH -> RK -> RMrot -> RF
]


def load_motion(pt_path: str) -> np.ndarray:
    """
    Load motion data from .pt file (PyTorch tensor).

    HumanML3D format: (frames, 22, 3) where last dimension is (x, y, z)
    In HumanML3D: Y is vertical (up), X and Z are horizontal
    For proper 3D visualization: we'll map Y -> Z (vertical), X -> X, Z -> Y

    Returns numpy array for matplotlib visualization.
    """
    # Load PyTorch tensor and convert to numpy for visualization
    motion_tensor = torch.load(pt_path, map_location="cpu")
    motion = motion_tensor.numpy()

    print(f"Loaded motion: {motion.shape}")
    print(f"  Frames: {motion.shape[0]}")
    print(f"  Joints: {motion.shape[1]}")
    print(f"  Dimensions: {motion.shape[2]}")

    # Remap axes: HumanML3D (x, y, z) -> Visualization (x, z, y)
    # This makes Y axis (vertical in HumanML3D) become Z axis (vertical in plot)
    motion_remapped = motion.copy()
    motion_remapped[:, :, [0, 1, 2]] = motion[:, :, [0, 2, 1]]  # x, z, y <- x, y, z

    return motion_remapped


def setup_3d_plot():
    """Set up the 3D plot with proper viewing angle."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig, ax


def update_frame(frame_idx: int, motion: np.ndarray, ax, lines: list, points: list):
    """Update function for animation."""
    ax.clear()

    # Get current frame
    frame = motion[frame_idx]

    # Set consistent axis limits based on all frames
    all_coords = motion.reshape(-1, 3)
    margin = 0.5
    x_range = [all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin]
    y_range = [all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin]
    z_range = [0, all_coords[:, 2].max() + margin]  # Z starts at ground (0)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)

    # Set labels and title
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z (Height)", fontsize=10)
    ax.set_title(f"Frame {frame_idx + 1}/{len(motion)}", fontsize=14, pad=20)

    # Set viewing angle (slightly elevated, rotated for better view)
    ax.view_init(elev=15, azim=45)

    # Draw ground plane at z=0
    xx, yy = np.meshgrid(
        np.linspace(x_range[0], x_range[1], 2), np.linspace(y_range[0], y_range[1], 2)
    )
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color="gray")

    # Plot skeleton bones with different colors for different parts
    colors = ["red", "blue", "green", "cyan", "magenta"]
    for chain_idx, chain in enumerate(SKELETON_CHAINS):
        color = colors[chain_idx % len(colors)]
        for i in range(len(chain) - 1):
            j1, j2 = chain[i], chain[i + 1]
            if j1 < len(frame) and j2 < len(frame):
                xs = [frame[j1, 0], frame[j2, 0]]
                ys = [frame[j1, 1], frame[j2, 1]]
                zs = [frame[j1, 2], frame[j2, 2]]
                linewidth = 4.0 if chain_idx == 0 else 3.0  # Thicker for body
                ax.plot(xs, ys, zs, color=color, linewidth=linewidth, alpha=0.8)

    # Plot joints (darker red)
    ax.scatter(
        frame[:, 0],
        frame[:, 1],
        frame[:, 2],
        c="darkred",
        marker="o",
        s=50,
        alpha=0.9,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add grid
    ax.grid(True, alpha=0.3)

    return (ax,)


def create_video_from_joints(
    joints: torch.Tensor | np.ndarray, output_path: str, fps: int = 20
) -> str:
    """
    Create 3D skeleton animation directly from joint tensor or array.

    Args:
        joints: Joint positions as torch.Tensor or np.ndarray (frames, 22, 3)
        output_path: Path to save video
        fps: Frames per second for the video

    Returns:
        Path to output video
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()

    # Remap axes for visualization (same as load_motion)
    motion = joints.copy()
    motion[:, :, [0, 1, 2]] = joints[:, :, [0, 2, 1]]  # x, z, y <- x, y, z

    # Set up plot
    fig, ax = setup_3d_plot()
    lines, points = [], []

    # Create animation
    anim = FuncAnimation(
        fig,
        update_frame,
        frames=len(motion),
        fargs=(motion, ax, lines, points),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    # Save video using FFMpeg
    writer = FFMpegWriter(fps=fps, bitrate=1800, codec="libx264")
    anim.save(str(output_path), writer=writer, dpi=100)

    plt.close(fig)
    return str(output_path)


def visualize_motion(
    pt_path: str, output_path: str | None = None, fps: int = 20, show: bool = False
) -> str:
    """
    Visualize motion from .pt file (PyTorch tensor).

    Args:
        pt_path: Path to .pt motion file
        output_path: Path to save video (if None, will auto-generate)
        fps: Frames per second for the video
        show: If True, display the animation in a window

    Returns:
        Path to the generated video file
    """
    # Load motion data (converts to numpy internally for matplotlib)
    motion = load_motion(pt_path)

    # Create output path if not specified
    if output_path is None:
        output_path = Path(pt_path).with_suffix(".mp4")
    else:
        output_path = Path(output_path)

    print(f"\nCreating animation with {fps} FPS...")

    # Set up plot
    fig, ax = setup_3d_plot()
    lines, points = [], []

    # Create animation
    anim = FuncAnimation(
        fig,
        update_frame,
        frames=len(motion),
        fargs=(motion, ax, lines, points),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    # Save video using FFMpeg
    print(f"Saving video to: {output_path}")
    writer = FFMpegWriter(fps=fps, bitrate=1800, codec="libx264")
    anim.save(str(output_path), writer=writer, dpi=100)
    print("✓ Video saved successfully!")

    # Show animation if requested
    if show:
        plt.show()

    plt.close(fig)
    return str(output_path)


def main() -> int:
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="Visualize HumanML3D motion data as 3D skeleton animation"
    )
    parser.add_argument("input", type=str, help="Path to input .pt motion file")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output video file (default: input_name.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for output video (default: 20)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the animation in a window (in addition to saving)",
    )

    args = parser.parse_args()

    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Visualize the motion
    try:
        output_path = visualize_motion(
            args.input, output_path=args.output, fps=args.fps, show=args.show
        )
        print(f"\n✓ Done! Video saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
