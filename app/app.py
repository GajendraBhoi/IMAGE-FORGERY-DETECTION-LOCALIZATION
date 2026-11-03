from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import THRESHOLD, ModelLoadError, predict_image

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _create_overlay(image: Image.Image, binary_mask: np.ndarray) -> Image.Image:
    original = image.convert("RGB")
    resized_mask = Image.fromarray(binary_mask).resize(
        original.size, Image.Resampling.NEAREST
    )
    original_pixels = np.asarray(original).copy()
    manipulated = np.asarray(resized_mask) > 0
    red = np.array([255, 48, 48], dtype=np.float32)
    original_pixels[manipulated] = (
        original_pixels[manipulated].astype(np.float32) * 0.5 + red * 0.5
    ).astype(np.uint8)
    return Image.fromarray(original_pixels)


def _run_prediction(
    image: Image.Image | None,
) -> tuple[
    Image.Image | None,
    Image.Image | None,
    Image.Image | None,
    Image.Image | None,
    str,
]:
    if image is None:
        raise gr.Error("Upload an image before starting detection.")
    if not isinstance(image, Image.Image):
        raise gr.Error("Upload a supported image file.")

    try:
        original = image.convert("RGB")
        binary_mask, probability_map = predict_image(original)
    except ModelLoadError as exc:
        LOGGER.exception("F²Mamba checkpoint could not be loaded")
        raise gr.Error(f"Model unavailable: {exc}") from exc
    except (OSError, TypeError, UnidentifiedImageError, ValueError) as exc:
        LOGGER.exception("Uploaded image could not be processed")
        raise gr.Error(f"Invalid or unsupported image: {exc}") from exc
    except RuntimeError as exc:
        LOGGER.exception("F²Mamba prediction failed")
        raise gr.Error(f"Prediction failed: {exc}") from exc

    mask_image = Image.fromarray(binary_mask).resize(
        original.size, Image.Resampling.NEAREST
    )
    probability_image = Image.fromarray(
        np.clip(probability_map * 255.0, 0, 255).astype(np.uint8)
    ).resize(original.size, Image.Resampling.BILINEAR)
    overlay = _create_overlay(original, binary_mask)
    detected_area = float((binary_mask > 0).mean() * 100.0)
    mean_probability = float(probability_map.mean())
    information = (
        f"**Detected area:** {detected_area:.2f}% of the image  \n"
        f"**Mean pixel probability:** {mean_probability:.3f}  \n"
        f"**Decision threshold:** {THRESHOLD:.2f}  \n"
        f"**Device:** {get_device_label()}"
    )
    return original, mask_image, probability_image, overlay, information


def get_device_label() -> str:
    from src.inference import get_inference_engine

    return str(get_inference_engine().device).upper()


CSS = """
.gradio-container {max-width: 1200px !important; margin: 0 auto !important;}
.hero {padding: 1.5rem 1rem; text-align: center;}
.hero h1 {color: #123b70; margin-bottom: .35rem;}
.hero p {color: #53657d; font-size: 1.05rem;}
.section-card {border: 1px solid #dce7f3; border-radius: 16px; padding: 1rem;}
@media (max-width: 700px) {
  .gradio-container {padding: .5rem !important;}
  .hero {padding: 1rem .25rem;}
}
"""

with gr.Blocks(
    title="F²Mamba – Image Forgery Localization",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky", neutral_hue="slate"),
    css=CSS,
) as demo:
    gr.Markdown(
        """
        <div class="hero">
          <h1>F²Mamba – Image Forgery Localization</h1>
          <p>Research prototype for pixel-level localization of manipulated image regions.</p>
        </div>
        """
    )
    gr.Markdown(
        "Upload an image to generate a probability map and a binary localization mask. "
        "Red overlay pixels indicate regions above the configured 0.35 threshold."
    )

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["section-card"]):
            image_input = gr.Image(
                type="pil",
                label="Upload an image",
                sources=["upload"],
                height=360,
            )
            detect_button = gr.Button("Detect Forgery", variant="primary", size="lg")
        with gr.Column(scale=2):
            with gr.Row():
                original_output = gr.Image(label="Original image")
                mask_output = gr.Image(label="Predicted binary mask")
            with gr.Row():
                probability_output = gr.Image(label="Probability map")
                overlay_output = gr.Image(label="Detected regions in red")
            prediction_info = gr.Markdown(
                "Prediction details will appear here after analysis."
            )

    detect_button.click(
        fn=_run_prediction,
        inputs=image_input,
        outputs=[
            original_output,
            mask_output,
            probability_output,
            overlay_output,
            prediction_info,
        ],
    )
    gr.Markdown(
        "_Research use only. The output is a model prediction and should not be "
        "treated as a definitive authenticity judgment._"
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
