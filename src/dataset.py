from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = (512, 512)


def _open_image(source: Image.Image | str | Path) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.copy()
    with Image.open(source) as image:
        return image.copy()


def preprocess_image(
    image: Image.Image | str | Path,
) -> torch.Tensor:
    rgb_image = _open_image(image).convert("RGB")
    rgb_image = rgb_image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    pixels = np.asarray(rgb_image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


def preprocess_mask(mask: Image.Image | str | Path) -> torch.Tensor:
    grayscale_mask = _open_image(mask).convert("L")
    grayscale_mask = grayscale_mask.resize(IMAGE_SIZE, Image.Resampling.NEAREST)
    pixels = np.asarray(grayscale_mask)
    binary_mask = (pixels >= 128).astype(np.float32)
    return torch.from_numpy(binary_mask).unsqueeze(0)


class ForgeryDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, pairs: Sequence[tuple[str | Path, str | Path]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        return preprocess_image(image_path), preprocess_mask(mask_path)
