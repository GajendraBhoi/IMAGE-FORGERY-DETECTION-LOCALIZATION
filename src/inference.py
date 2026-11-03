from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import preprocess_image
from src.model import DevelopedF2Mamba

LOGGER = logging.getLogger(__name__)
THRESHOLD = 0.35
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "models" / "best_defacto_finetuned.pth"
)


class ModelLoadError(RuntimeError):
    pass


def _resolve_checkpoint() -> Path:
    local_path = os.environ.get("F2MAMBA_CHECKPOINT_PATH")
    if local_path:
        checkpoint = Path(local_path).expanduser()
        if not checkpoint.is_file():
            raise ModelLoadError(
                f"F2MAMBA_CHECKPOINT_PATH does not point to a file: {checkpoint}"
            )
        return checkpoint

    repo_id = os.environ.get("F2MAMBA_HF_REPO_ID")
    if repo_id:
        filename = os.environ.get("F2MAMBA_HF_FILENAME", "best_defacto_finetuned.pth")
        try:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Could not download {filename!r} from Hugging Face repository "
                f"{repo_id!r}: {exc}"
            ) from exc
        return Path(downloaded_path)

    if DEFAULT_CHECKPOINT.is_file():
        return DEFAULT_CHECKPOINT

    raise ModelLoadError(
        f"Checkpoint not found at {DEFAULT_CHECKPOINT}. Set "
        "F2MAMBA_CHECKPOINT_PATH to a local checkpoint or "
        "F2MAMBA_HF_REPO_ID to a Hugging Face repository."
    )


class InferenceEngine:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = _resolve_checkpoint()

        try:
            model = DevelopedF2Mamba(num_classes=1)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            if not isinstance(state_dict, Mapping):
                raise TypeError("Checkpoint must contain a model state dictionary.")
            model.load_state_dict(state_dict, strict=True)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            raise ModelLoadError(
                f"Could not load a compatible F²Mamba checkpoint from "
                f"{checkpoint_path}: {exc}"
            ) from exc

        self.model = model.to(self.device)
        self.model.eval()
        LOGGER.info(
            "Loaded F²Mamba checkpoint from %s on %s",
            checkpoint_path,
            self.device,
        )

    def predict(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(image, Image.Image):
            raise TypeError("Provide a valid image to analyze.")

        input_tensor = preprocess_image(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            prediction = self.model(input_tensor)

        if prediction.ndim != 4 or prediction.shape[:2] != (1, 1):
            raise RuntimeError(
                f"Expected one 512x512 prediction map, got {tuple(prediction.shape)}."
            )

        probability_map = prediction[0, 0].float().cpu().numpy()
        binary_mask = (probability_map >= THRESHOLD).astype(np.uint8) * 255
        return binary_mask, probability_map


_ENGINE: InferenceEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_inference_engine() -> InferenceEngine:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = InferenceEngine()
    return _ENGINE


def predict_image(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    return get_inference_engine().predict(image)
