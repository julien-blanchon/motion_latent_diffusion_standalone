from pathlib import Path
import gradio as gr
import torch
from datetime import datetime
import tempfile
from tqdm import tqdm
from textwrap import dedent
import spaces


from motion_latent_diffusion_standalone import MotionLatentDiffusionModel
from visualize import create_video_from_joints


model = MotionLatentDiffusionModel(
    vae_repo_id="blanchon/motion-latent-diffusion-standalone-vae",
    denoiser_repo_id="blanchon/motion-latent-diffusion-standalone-denoiser",
    text_encoder_repo_id="openai/clip-vit-large-patch14",
)
model.to("cuda")
model.eval()
model.requires_grad_(False)


@spaces.gpu()
def generate_motion(
    text_prompt: str, motion_length: int, progress=gr.Progress(track_tqdm=True)
) -> tuple[Path, str, Path]:
    try:
        # Create temporary files
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_{timestamp}"

        pt_path = Path(temp_dir) / f"{filename}.pt"
        video_path = Path(temp_dir) / f"{filename}.mp4"

        print("🎬 Generating motion...")
        with tqdm(
            total=motion_length,
            desc="Generating motion",
            # disable=not progress.is_tracked(),
        ) as pbar:

            def callback_on_step_end(i: int, latents: torch.Tensor):
                pbar.update(i)

            # Generate motion (returns PyTorch tensor)
            joints, latent = model.generate(
                text_prompt,
                motion_length,
                return_latent=True,
                callback_on_step_end=callback_on_step_end,
            )

        # Save motion data as PyTorch tensor
        torch.save(joints, pt_path)

        print("🎥 Creating visualization...")

        # Create video visualization
        video_path = create_video_from_joints(joints, video_path.as_posix(), fps=20)

        print("✅ Done!")

        # Generate info text
        info_text = dedent("""
            ✅ **Generation Complete!**

            **Prompt:** {text_prompt}
            **Motion Length:** {motion_length} frames ({motion_length / 20:.1f}s at 20fps)
            **Output Shape:** {joints.shape} (frames × joints × coords)

            The video shows a 3D skeleton performing the motion. 
            You can download both the video and the raw motion data below.
        """)

        return video_path, info_text, pt_path.as_posix()

    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        import traceback

        traceback.print_exc()
        return None, error_msg, None


def create_example_prompts():
    """Return example prompts for the interface"""
    return [
        ["a person walks forward slowly", 80],
        ["jumping up and down", 100],
        ["a person waves hello", 60],
        ["running in place", 100],
        ["a person does jumping jacks", 120],
        ["someone performs a cartwheel", 140],
        ["walking backwards carefully", 90],
        ["a person stretches their arms", 80],
    ]


with gr.Blocks(title="MLD Text-to-Motion Generator", theme=gr.themes.Soft()) as demo:
    # Header
    gr.Markdown("""
    # 🎬 MLD Text-to-Motion Generator
    
    Generate realistic human motion animations from text descriptions! 
    Powered by Motion Latent Diffusion (MLD).
    
    ### 💡 Tips for Best Results:
    - Be specific: "a person walks forward slowly" works better than just "walking"
    - Use present tense: "walks" or "is walking"
    - Describe single continuous actions
    - Recommended length: 40-60 frames for short actions, 80-120 for walking/running
    """)

    with gr.Row():
        # Left column - Inputs
        with gr.Column(scale=1):
            gr.Markdown("## 📝 Input")

            text_input = gr.Textbox(
                label="Text Prompt",
                placeholder="Enter motion description (e.g., 'a person walks forward slowly')",
                lines=3,
                value="a person walks forward",
            )

            with gr.Row():
                length_slider = gr.Slider(
                    minimum=16,
                    maximum=196,
                    value=100,
                    step=1,
                    label="Motion Length (frames)",
                    info="20 frames = 1 second",
                )

            generate_btn = gr.Button("🎬 Generate Motion", variant="primary", size="lg")

            gr.Markdown("### 📚 Example Prompts")
            gr.Examples(
                examples=create_example_prompts(),
                inputs=[text_input, length_slider],
                label=None,
            )

        # Right column - Outputs
        with gr.Column(scale=1):
            gr.Markdown("## 🎥 Output")

            info_output = gr.Markdown(
                "Generate a motion to see the results here.",
                elem_classes=["output-info"],
            )

            video_output = gr.Video(
                label="Generated Motion Video",
                elem_classes=["output-video"],
                autoplay=True,
                show_share_button=True,
            )

            with gr.Row():
                pt_download = gr.File(label="Download Motion Data (.pt)", visible=False)

    # Footer
    gr.Markdown(
        dedent("""
        ---
        ### ℹ️ About
        
        **Motion Latent Diffusion (MLD)** generates 3D human motion by:
        1. Encoding text with CLIP
        2. Generating motion in latent space via diffusion (50 steps)
        3. Decoding to 3D joint positions (22 joints)
        4. Visualizing as a 3D skeleton animation
        
        **Citation:** Chen et al., "Executing your Commands via Motion Diffusion in Latent Space", CVPR 2023
        
        **Repository:** [motion-latent-diffusion](https://github.com/ChenFengYe/motion-latent-diffusion)
        """)
    )

    # Event handlers
    def generate_and_update(text, length):
        video, info, pt = generate_motion(text, length)
        if pt:
            return video, info, gr.update(value=pt, visible=True)
        return video, info, gr.update(visible=False)

    generate_btn.click(
        fn=generate_and_update,
        inputs=[text_input, length_slider],
        outputs=[video_output, info_output, pt_download],
    )


demo.launch(
    server_name="0.0.0.0",  # Allow external access
    server_port=7860,
    share=False,
    show_error=True,
)
