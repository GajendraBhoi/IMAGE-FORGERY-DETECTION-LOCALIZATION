---
title: F²Mamba – Image Forgery Localization
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app/app.py
pinned: false
---

# F²Mamba – Image Forgery Localization

F²Mamba is a research prototype for locating manipulated regions in an image. It
combines hierarchical ResNet34 features and a pure-PyTorch S6 state-space
bottleneck with forensic noise features. The Gradio app returns a binary
localization mask, a pixel probability map, and a red overlay.

The deployed application needs only the trained checkpoint and an uploaded
image; it does not access Kaggle datasets.

## Problem statement

Image splicing and other editing operations can leave localized traces that are
hard to identify by visual inspection alone. F²Mamba predicts a per-pixel
forgery probability map to help researchers inspect candidate manipulated
regions. Its predictions are not a definitive authenticity judgment.

## F²Mamba architecture

The implementation in `src/model.py` follows the final `DevelopedF2Mamba`
definition used to fine-tune the supplied checkpoint:

- **ResNet34 encoder:** `timm` ResNet34 in `features_only` mode produces five
  hierarchical feature maps for a 512 × 512 RGB input.
- **S6/Mamba bottleneck:** `PurePyTorchS6Block` uses `d_model=512` and
  `d_state=16`. The deepest feature map is flattened into a sequence, processed
  by the selective state-space recurrence in float32, then reshaped.
- **FFA forensic branch:** `StreamlinedFFA` concatenates per-channel fixed SRM
  5 × 5 filters, a learned 3-to-4-channel Bayar convolution, and per-channel
  fixed 3 × 3 Laplacian filters. Its fusion head maps the ten feature channels
  through 16 channels to one forensic prediction channel. The notebook's
  runtime Bayar-filter normalization fix is retained in the forward pass.
- **Decoder:** five transposed-convolution upsampling stages use concatenated
  ResNet skip features and convolution/BatchNorm/ReLU blocks.
- **Confidence-gated fusion:** a learned gate blends the sigmoid decoder
  prediction with the sigmoid FFA prediction into a one-channel probability
  map. The model returns probabilities, not logits.

The application resizes inputs to 512 × 512, converts them to RGB, scales
channels to [0, 1] without ImageNet mean/std normalization, and thresholds
probabilities at **0.35**.

## Datasets

The research workflow uses four datasets:

- CASIA v2.0
- IMD2020
- COVERAGE
- DEFACTO-Splicing

Dataset paths in the research notebook are specific to its training environment
and are intentionally not used by deployed inference.

## Training configuration

The notebook trains a multi-dataset master model and then fine-tunes it on
DEFACTO-Splicing. The final DEFACTO fine-tuning configuration is:

| Setting | Value |
|---|---:|
| Input and mask size | 512 × 512 |
| Master multi-dataset training | 20 epochs, batch size 16, AdamW learning rate 1.5e-4 |
| DEFACTO-Splicing fine-tuning | 3 epochs, batch size 8 |
| Fine-tuning optimizer | AdamW, learning rate 5e-5, weight decay 1e-4 |
| Fine-tuning loss | 0.5 Dice + 0.5 focal |
| Mixed precision | CUDA autocast and gradient scaling during training |
| Checkpoint selection | Best validation pixel-level F1 |
| Inference threshold | 0.35 |

During fine-tuning, the training split combines CASIA v2.0, IMD2020, COVERAGE,
and 70% of the DEFACTO-Splicing pairs; the remaining DEFACTO-Splicing pairs are
split equally between validation and test.

## Training results

The selected DEFACTO-Splicing fine-tuned checkpoint reported validation
pixel-level F1 of **0.8401** and IoU of **0.7243** at threshold 0.35.

## Test evaluation results

The following are the supplied final test results at threshold 0.35.

### Test set: aggregate pixel-level metrics

Counts are aggregated across pixels from the evaluated images before
calculating precision, recall, F1, and IoU.

| Metric | Score |
|---|---:|
| Precision | 0.8160 |
| Recall | 0.8232 |
| Pixel-level F1 | 0.8196 |
| Pixel-level IoU | 0.6943 |

### Test set: mean per-image metrics

F1 and IoU are calculated independently for each image, then averaged. These
are distinct from the aggregate pixel-level scores above.

| Metric | Score |
|---|---:|
| Mean per-image F1 | 0.6480 |
| Mean per-image IoU | 0.5563 |

## Installation

Use Python 3.10 or newer. From the repository root, install the inference and
application dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The installed PyTorch build should match the host accelerator. For CUDA
deployment, select the appropriate PyTorch wheel for the target CUDA version
from the official PyTorch installation instructions.

## Run locally

By default, the app looks for
`models/best_defacto_finetuned.pth` relative to the project root:

```bash
python app/app.py
```

Open `http://localhost:7860` in a browser. To use a checkpoint at another local
path, set `F2MAMBA_CHECKPOINT_PATH` before starting the app:

```powershell
$env:F2MAMBA_CHECKPOINT_PATH = "C:\path\to\best_defacto_finetuned.pth"
python app\app.py
```

To load the checkpoint from Hugging Face Hub instead, configure:

```text
F2MAMBA_HF_REPO_ID=your-account/your-model-repository
F2MAMBA_HF_FILENAME=best_defacto_finetuned.pth
```

`F2MAMBA_HF_FILENAME` is optional and defaults to
`best_defacto_finetuned.pth`. For a private model repository, add `HF_TOKEN` as
a secret in the hosting environment. An explicit local
`F2MAMBA_CHECKPOINT_PATH` takes precedence over the Hub configuration. If
neither is set, the app uses the default local checkpoint path.

The model is loaded once per app process and reused for subsequent predictions.
It selects CUDA when available and otherwise runs on CPU. The model loading
error is reported in the interface if the configured checkpoint is missing or
incompatible.

## Deploy on Hugging Face Spaces

The README metadata configures this repository as a Gradio Space using
`app/app.py`. Add the checkpoint to a Hugging Face model repository rather than
committing it to GitHub or the Space. Set the Space variable
`F2MAMBA_HF_REPO_ID` and, if needed, `F2MAMBA_HF_FILENAME`; add `HF_TOKEN` as a
Space secret only for a private model repository.

## Example screenshots

Screenshots are not included yet. Add captures of the upload screen and a
prediction (mask, probability map, and overlay) under
`assets/screenshots/` when available.

## Future improvements

- Add a curated set of representative examples and failure cases.
- Evaluate generalization across dataset domains and image transformations.
- Add batch inference and downloadable prediction artifacts.
- Explore calibrated uncertainty estimates and explainability views.
- Expand deployment monitoring and model-version reporting.
