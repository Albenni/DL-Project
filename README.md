# Swirl Correction (Keras) — Two-Stage U-Net under 4M Params

> Identify and correct the portion of an input image affected by a moderate **swirl** effect.  
> The swirl **center**, **radius**, and **intensity** vary per image.

## Highlights
- **Keras / TensorFlow 2** implementation end-to-end.
- **Parameter budget ≤ 4M** across the models (see the snippet below to verify on your machine).
- **Pretrained weights via `gdown`** for quick inference.
- Self-contained **Jupyter notebook**: `Swirl_correction.ipynb`.

## Method Overview
The pipeline uses **two lightweight U-Nets** trained at **128×128** resolution:

1) **Mask Estimation U-Net**  
   Predicts a 1-channel mask of the region affected by the swirl.
   - Encoder: 3 downsampling blocks with filters `[16, 32, 64]` (with `base_filters=16`).
   - Bottleneck: two `3×3 Conv` layers (128 filters).
   - Decoder: 3 upsampling blocks (transpose conv) with skip connections.
   - Output: `1×1 Conv` + `sigmoid` → `H×W×1` mask.

2) **Reconstruction U-Net**  
   Repairs the corrupted area, conditioned on the predicted mask.
   - Input: original RGB **concatenated** with mask → `H×W×4`.
   - Encoder: 3 downsampling blocks with filters `[64, 128, 256]`.
   - Bottleneck: two `3×3 Conv` layers (256 filters).
   - Decoder: 3 upsampling blocks with skip connections, filters `[128, 64, 64]`.
   - Output: `1×1 Conv` + `sigmoid` → restored RGB image.

**Losses**  
- **Mask U-Net**: binary cross-entropy (BCE).  
- **Reconstruction U-Net**: a mix of pixel and perceptual similarity:  
  `0.9 * MSE + 0.1 * (1 - mean SSIM)`.

> The reconstruction model **freezes** the mask model and uses its prediction as input.

## Data
Training uses **`tf_flowers`** from **TensorFlow Datasets**. Swirl-corrupted inputs and ground-truth masks are **synthetically generated** with `skimage.transform.swirl` (randomized center, radius, and strength).

- Clean target: the original (resized) image.
- Defected input: swirl-warped version.
- Mask target: disk-like / region delineating the warp.

> This makes the project easy to reproduce without external data downloads.

## Quickstart

### Environment
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install --upgrade pip

# Core
pip install tensorflow tensorflow-datasets scikit-image numpy matplotlib

# Tools
pip install gdown
```

## Known limitations
- Extremely strong or complex swirl artifacts may require a deeper model (which could threaten the 4M cap).
- Generalization depends on how representative the training data/augmentations are.

## Why this approach?
- Keras makes the model concise and easy to iterate on.
- The ≤ 4M parameter cap encourages efficient design and fast deployment.
- `gdown` weights provide a frictionless path for reviewers to run inference.

## Acknowledgements
- TensorFlow / Keras  
- NumPy, scikit-image, Matplotlib

## License
MIT (see `LICENSE`).


