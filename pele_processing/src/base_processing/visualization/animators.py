"""
Animation generation for the Pele processing system.
"""
import re
import time
from typing import List, Optional, Union, Dict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

from ..core.interfaces import AnimationBuilder
from ..core.exceptions import AnimationError


class FrameAnimator(AnimationBuilder):
    """Creates animations from frame sequences."""

    def __init__(self):
        self.supported_formats = ['gif', 'mp4', 'avi', 'mov']

    def create_animation(self, frame_directory: Path, output_path: Path,
                         frame_rate: float = 5.0, format: str = 'gif') -> None:
        """Create animation from frames."""
        if format not in self.supported_formats:
            raise AnimationError("format", f"Unsupported format: {format}")

        # Get sorted frame files
        frame_files = self._get_sorted_frames(frame_directory)
        if not frame_files:
            raise AnimationError("frames", "No frame files found")

        try:
            if format == 'gif':
                self._create_gif_animation(frame_files, output_path, frame_rate)
            else:
                self._create_video_animation(frame_files, output_path, frame_rate, format)

        except Exception as e:
            raise AnimationError("creation", str(e))

    def _get_sorted_frames(self, directory: Path) -> List[Path]:
        """Get frame files sorted by dataset number."""
        frame_files = list(directory.glob("*.png"))

        def extract_number(filename: str) -> int:
            match = re.search(r'plt(\d+)', filename)
            return int(match.group(1)) if match else 0

        return sorted(frame_files, key=lambda x: extract_number(x.name))

    def _create_gif_animation(self, frame_files: List[Path], output_path: Path,
                              frame_rate: float) -> None:
        """Create GIF animation using Pillow."""
        try:
            from PIL import Image
        except ImportError:
            raise AnimationError("dependency", "Pillow required for GIF creation")

        images = []
        for frame_file in frame_files:
            img = Image.open(frame_file)
            images.append(img)

        duration = int(1000 / frame_rate)  # milliseconds per frame
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )

    def _create_video_animation(self, frame_files: List[Path], output_path: Path,
                                frame_rate: float, format: str) -> None:
        """Create video animation using matplotlib."""
        if not frame_files:
            return

        # Load first frame to get dimensions
        first_frame = mpimg.imread(frame_files[0])
        fig = plt.figure(figsize=(first_frame.shape[1] / 100, first_frame.shape[0] / 100), dpi=100)
        plt.axis('off')

        img_display = plt.imshow(first_frame)

        def update_frame(frame_idx):
            if frame_idx < len(frame_files):
                img = mpimg.imread(frame_files[frame_idx])
                img_display.set_array(img)
            return [img_display]

        anim = animation.FuncAnimation(
            fig, update_frame, frames=len(frame_files),
            interval=1000 / frame_rate, blit=True
        )

        # Choose writer based on format
        writers = {
            'mp4': 'ffmpeg',
            'avi': 'ffmpeg',
            'mov': 'ffmpeg'
        }

        writer_name = writers.get(format, 'ffmpeg')
        try:
            writer = animation.writers[writer_name](fps=frame_rate, bitrate=1800)
            anim.save(output_path, writer=writer)
        except Exception as e:
            # Fallback to PillowWriter for gif
            if format == 'gif':
                writer = animation.PillowWriter(fps=frame_rate)
                anim.save(output_path, writer=writer)
            else:
                raise

        plt.close(fig)


class BatchAnimator:
    """Creates multiple animations in batch."""

    def __init__(self, animator: AnimationBuilder = None):
        self.animator = animator or FrameAnimator()

    def create_all_animations(self, frames_base_dir: Path, output_dir: Path,
                              frame_rate: float = 5.0, formats: List[str] = None) -> Dict[str, bool]:
        """Create animations for all field directories."""
        formats = formats or ['gif']
        results = {}

        # Find all frame directories
        frame_dirs = [d for d in frames_base_dir.iterdir()
                      if d.is_dir() and d.name.endswith('_frames')]

        for frame_dir in frame_dirs:
            field_name = frame_dir.name.replace('_frames', '')

            for fmt in formats:
                output_file = output_dir / f"{field_name}_animation.{fmt}"

                try:
                    self.animator.create_animation(
                        frame_dir, output_file, frame_rate, fmt
                    )
                    results[f"{field_name}_{fmt}"] = True
                except Exception as e:
                    results[f"{field_name}_{fmt}"] = False
                    print(f"Failed to create {field_name} animation: {e}")

        return results

    def create_comparison_animation(self, frame_dirs: Dict[str, Path],
                                    output_path: Path, frame_rate: float = 5.0) -> None:
        """Create side-by-side comparison animation."""
        # Get common frame files
        all_frames = {}
        for name, directory in frame_dirs.items():
            frames = self._get_sorted_frames(directory)
            all_frames[name] = frames

        if not all_frames:
            raise AnimationError("frames", "No frames found for comparison")

        # Find minimum number of frames
        min_frames = min(len(frames) for frames in all_frames.values())

        # Create comparison frames
        comparison_dir = output_path.parent / f"comparison_frames_{int(time.time())}"
        comparison_dir.mkdir(exist_ok=True)

        try:
            for i in range(min_frames):
                self._create_comparison_frame(
                    {name: frames[i] for name, frames in all_frames.items()},
                    comparison_dir / f"comparison_frame_{i:06d}.png"
                )

            # Create animation from comparison frames
            self.animator.create_animation(comparison_dir, output_path, frame_rate)

        finally:
            # Cleanup comparison frames
            import shutil
            shutil.rmtree(comparison_dir, ignore_errors=True)

    def _get_sorted_frames(self, directory: Path) -> List[Path]:
        """Get sorted frame files."""
        frame_files = list(directory.glob("*.png"))

        def extract_number(filename: str) -> int:
            match = re.search(r'plt(\d+)', filename)
            return int(match.group(1)) if match else 0

        return sorted(frame_files, key=lambda x: extract_number(x.name))

    def _create_comparison_frame(self, frame_paths: Dict[str, Path],
                                 output_path: Path) -> None:
        """Create side-by-side comparison frame."""
        import matplotlib.pyplot as plt

        n_plots = len(frame_paths)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

        if n_plots == 1:
            axes = [axes]

        for ax, (name, frame_path) in zip(axes, frame_paths.items()):
            img = mpimg.imread(frame_path)
            ax.imshow(img)
            ax.set_title(name)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


class InteractiveAnimator:
    """Creates interactive animations with controls."""

    def __init__(self):
        self.current_frame = 0
        self.playing = False

    def create_interactive_plot(self, frame_files: List[Path]) -> None:
        """Create interactive matplotlib animation with controls."""
        if not frame_files:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Load first frame
        first_img = mpimg.imread(frame_files[0])
        img_display = ax.imshow(first_img)
        ax.axis('off')

        # Add controls
        from matplotlib.widgets import Button, Slider

        # Create control panel
        ax_controls = plt.axes([0.1, 0.02, 0.8, 0.1])
        ax_controls.set_xlim(0, 10)
        ax_controls.set_ylim(0, 2)
        ax_controls.axis('off')

        # Play/pause button
        ax_play = plt.axes([0.1, 0.02, 0.1, 0.04])
        btn_play = Button(ax_play, 'Play/Pause')

        # Frame slider
        ax_slider = plt.axes([0.25, 0.02, 0.5, 0.04])
        slider = Slider(ax_slider, 'Frame', 0, len(frame_files) - 1,
                        valinit=0, valfmt='%d')

        def update_frame(frame_idx):
            frame_idx = int(frame_idx)
            if 0 <= frame_idx < len(frame_files):
                img = mpimg.imread(frame_files[frame_idx])
                img_display.set_array(img)
                self.current_frame = frame_idx
                fig.canvas.draw()

        def toggle_play(event):
            self.playing = not self.playing

        slider.on_changed(update_frame)
        btn_play.on_clicked(toggle_play)

        plt.show()