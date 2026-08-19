"""
pipeline_utils.py
------------------
Reusable functions for the Biomedical Image Analysis pipeline assignment.

Modality: fluorescence microscopy (DAPI-style stained cell nuclei), synthetic
dataset with 256x256 RGB images, binary masks (0/255) and 16-bit instance
label maps, provided in nuclei_dataset/{train,val,test,test_corrupted}.

The functions here are organised by pipeline stage so that Tasks 1-4 in the
notebook can import them rather than repeating logic in every cell:
    - I/O & preprocessing        (Task 1)
    - Ollama (local LLM) client  (Task 1, 2, 4)
    - Classical feature pipeline (Task 2, 4)
    - U-Net model + training     (Task 3)
    - Evaluation metrics         (Task 3, 4)

Every function has a docstring explaining inputs/outputs so the notebook
cells that call them can stay short and readable.
"""

import os
import json
import time
import glob

import numpy as np
import pandas as pd
from PIL import Image

from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects, remove_small_holes, binary_opening, disk
)
from skimage.measure import label, regionprops_table

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# 1. I/O & preprocessing (Task 1)

def load_image_rgb(path):
    """Load a PNG as an RGB uint8 numpy array (H, W, 3)."""
    return np.array(Image.open(path).convert("RGB"))


def to_grayscale_resized(rgb_uint8, size=(256, 256)):
    """
    Convert an RGB uint8 image to a single-channel intensity map and resize
    to `size`. Returns a float32 array in [0, 1].

    IMPORTANT (modality-specific choice, verified against the dataset's own
    generator, make_dataset.py): this dataset is a DAPI-style stain where the
    generator writes its single-channel "intensity" array almost entirely
    into the BLUE channel (`rgb[...,2] = intensity`, `rgb[...,1] =
    intensity*0.35`, `rgb[...,0] = intensity*0.12`) before saving as RGB.
    A generic RGB->grayscale luminosity conversion (e.g. PIL's default "L"
    mode / ITU-R BT.601, weights ~0.299/0.587/0.114 for R/G/B) gives BLUE
    only 11.4% weight -- appropriate for natural photographs, but wrong here
    since blue carries essentially all of the real signal. That mismatch is
    directly checkable: metadata.csv's ground-truth `mean_intensity` column
    is computed by the generator directly from its pre-staining intensity
    array, and extracting the blue channel reproduces it almost exactly
    (e.g. train_000: ground truth 0.7475, blue channel 0.7481, vs. 0.262
    from a standard luminosity conversion -- verified numerically). We
    therefore use the blue channel directly as the "grayscale" signal for
    this dataset, rather than a generic luminosity formula.
    """
    img = Image.fromarray(rgb_uint8).resize(size, Image.BILINEAR)
    blue = np.asarray(img, dtype=np.float32)[:, :, 2] / 255.0
    return blue


def load_mask_binary(path, size=(256, 256)):
    """Load a ground-truth binary mask (0/255) as a {0,1} uint8 array, resized."""
    m = Image.open(path).convert("L").resize(size, Image.NEAREST)
    arr = (np.asarray(m) > 127).astype(np.uint8)
    return arr


def dataset_intensity_histogram(image_paths, bins=50):
    """
    Compute a pooled intensity histogram (grayscale, 0-1) across a list of
    image paths. Used for the Task 1 EDA. Returns (hist_counts, bin_edges).
    """
    all_pixels = []
    for p in image_paths:
        rgb = load_image_rgb(p)
        gray = to_grayscale_resized(rgb)
        all_pixels.append(gray.ravel())
    all_pixels = np.concatenate(all_pixels)
    counts, edges = np.histogram(all_pixels, bins=bins, range=(0, 1))
    return counts, edges


# ----
# 2. Local multimodal / text LLM client via Ollama (Task 1, 2, 4)
# ----
# NOTE: this uses the official `ollama` Python package (`from ollama import
# chat`), the same interface taught in Lab 2 (Lab2_multimodal_biomedical_colab.ipynb),
# rather than raw HTTP requests. Images are passed as a list of file paths
# under the `images` key of the message dict, exactly as in the lab.


def query_ollama(prompt, model="llama3.2-vision", image_path=None,
                  temperature=0.2):
    """
    Send a prompt (optionally with one image) to a local Ollama model and
    return the raw text response.

    Parameters
    ----------
    prompt : str              -- the text prompt (role + rules + format combined)
    model  : str               -- Ollama model tag, e.g. "llama3.2-vision" or "llama3.2"
    image_path : str or None   -- path to an image to attach (vision calls only)
    temperature : float        -- sampling temperature (kept low but >0 so we can
                                   demonstrate run-to-run variability, per Task 1)

    Returns
    -------
    str : the model's raw text response (response['message']['content']).

    Notes
    -----
    Requires Ollama running locally (`ollama serve`, or the Ollama desktop
    app which runs this in the background) and the model pulled beforehand,
    e.g. `ollama pull llama3.2-vision` and `ollama pull llama3.2`. This
    function is exercised via the OLLAMA_VISION_READY / OLLAMA_TEXT_READY
    guards in the notebook so it only runs for real when Ollama is reachable.

    Raises
    ------
    RuntimeError with a diagnostic message (instead of a raw ollama.ResponseError)
    if the model is pulled but fails to load -- this specifically catches the
    known "unknown model architecture: 'mllama'" failure some Ollama versions
    hit on llama3.2-vision (github.com/ollama/ollama/issues/16490, /16547),
    where the model downloads successfully but crashes on first use.
    """
    from ollama import chat
    from ollama import ResponseError

    message = {"role": "user", "content": prompt}
    if image_path is not None:
        message["images"] = [image_path]

    try:
        response = chat(model=model, messages=[message], options={"temperature": temperature})
    except ResponseError as e:
        if "unknown model architecture" in str(e).lower():
            raise RuntimeError(
                f"'{model}' is pulled but this Ollama install cannot run it "
                f"(architecture error: {e}). This matches a known, currently-unresolved "
                f"Ollama bug affecting llama3.2-vision on Ollama v0.30.0+ "
                f"(github.com/ollama/ollama/issues/16490, /16547). Fix: downgrade Ollama to "
                f"a pre-0.30.0 release, or set VISION_MODEL to an alternative such as "
                f"'llava:7b' or 'moondream' (both explicitly sanctioned as substitutes in "
                f"Lab 2) and re-run the Setup cell."
            ) from e
        raise
    return response["message"]["content"]


def extract_json_block(text):
    """
    Best-effort extraction of a JSON object from a model's text response.
    Models sometimes wrap JSON in markdown code fences (```json ... ```)
    even when told not to (as noted in Lab 2's `parse_model_json` helper),
    so we strip fences first, then fall back to locating the outermost
    {...} span. Returns a dict, or None if parsing fails either way.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# 3. Classical feature pipeline: Otsu -> morphology -> regionprops (Task 2, 4)

def segment_otsu(gray_float, min_size=8, hole_size=8, opening_radius=1):
    """
    Classical segmentation: Otsu threshold + morphological cleanup.

    Parameters
    ----------
    gray_float : (H, W) float32 array in [0, 1]
    min_size : minimum object size (pixels) to keep -- removes speckle noise
    hole_size : maximum hole area (pixels) to fill inside objects
    opening_radius : structuring element radius for binary opening
                      (separates loosely-touching objects, smooths edges)

    Returns
    -------
    binary : (H, W) uint8 {0, 1} cleaned foreground mask
    thresh : float, the Otsu threshold value used
    """
    thresh = threshold_otsu(gray_float)
    binary = gray_float > thresh
    binary = binary_opening(binary, footprint=disk(opening_radius))
    binary = remove_small_objects(binary, min_size=min_size)  # skimage>=0.26: 'min_size' kwarg still accepted as alias
    binary = remove_small_holes(binary, area_threshold=hole_size)
    return binary.astype(np.uint8), float(thresh)


def compute_region_table(binary_mask, intensity_image):
    """
    Label connected components in `binary_mask` and compute a per-object
    feature table using skimage.measure.regionprops_table.

    Returns a pandas DataFrame with one row per detected object and columns:
    label, area, eccentricity, solidity, mean_intensity, perimeter,
    major_axis_length, minor_axis_length.
    """
    labeled = label(binary_mask)
    props = ("label", "area", "eccentricity", "solidity", "mean_intensity",
              "perimeter", "major_axis_length", "minor_axis_length")
    table = regionprops_table(labeled, intensity_image=intensity_image,
                               properties=props)
    return pd.DataFrame(table), labeled


def summarize_region_table(df, density_thresholds=(15, 40), area_cv_clustered_threshold=0.9):
    """
    Turn a regionprops DataFrame into a short natural-language summary
    (numbers only, no image) for the Task 2 numbers-first LLM prompt.

    density_thresholds : (low_hi, mid_hi) object-count cutoffs used to
        separate sparse / normal / dense.
    area_cv_clustered_threshold : float
        This dataset's own README defines FOUR density regimes -- sparse,
        normal, dense, and clustered ("touching nuclei") -- not three.
        "Clustered" is a SHAPE property (touching/merged nuclei), not an
        object-count range: make_dataset.py's own count range for
        "clustered" (30-60) overlaps both "normal" (15-40) and "dense"
        (45-85), so object count alone cannot separate it out, and object
        count is all a naive sparse/normal/dense split can see.
        The signal that DOES carry information about merging is the
        variability of object AREA: when nuclei touch, Otsu merges them
        into connected components of very unequal size (a few large fused
        blobs alongside many normal single nuclei), which shows up as an
        elevated coefficient of variation (std/mean) of object area.
        Checked directly against this dataset's ground-truth density labels
        across all 80 training images: area CV > 0.9 catches 9/13 "clustered"
        images correctly, with a handful of false positives among dense
        images (mean area CV: sparse 0.45, normal 0.60, dense 0.81,
        clustered 1.07) -- a real but IMPERFECT classical signal, which is
        itself a fair point to discuss (Otsu + regionprops has no direct
        notion of "touching" instances, only indirect shape/area proxies
        for it).
    """
    n = len(df)
    if n == 0:
        return "No objects were detected after thresholding and cleanup.", "uncertain"

    lo_hi, mid_hi = density_thresholds
    mean_area = df["area"].mean()
    std_area = df["area"].std() if n > 1 else 0.0
    area_cv = (std_area / mean_area) if (n > 1 and mean_area > 0) else 0.0

    # Check area-CV (clustered) BEFORE the sparse count check: a heavily undercounted
    # clustered image (many merged nuclei -> few large connected components) can have a
    # LOW object count that would otherwise be caught by the sparse check first, even
    # though "sparse" (few, separated nuclei) never shows elevated area CV in practice
    # (verified: sparse images' area CV stayed well under this threshold in all 80
    # training images) -- so checking area CV first is safe and fixes that ordering bug.
    if area_cv > area_cv_clustered_threshold:
        density = "clustered"       # elevated area variability -> likely touching/merged nuclei
    elif n <= lo_hi:
        density = "sparse"
    elif n <= mid_hi:
        density = "normal"
    else:
        density = "dense"

    mean_ecc = df["eccentricity"].mean()
    mean_sol = df["solidity"].mean()
    mean_int = df["mean_intensity"].mean()

    summary = (
        f"Detected {n} connected components after Otsu thresholding and "
        f"morphological cleanup (approximate density class: {density}). "
        f"Object area: mean {mean_area:.1f} px^2, std {std_area:.1f} px^2 "
        f"(coefficient of variation {area_cv:.2f}). "
        f"Mean eccentricity {mean_ecc:.2f} (0=circle, 1=elongated line). "
        f"Mean solidity {mean_sol:.2f} (fraction of the convex hull filled; "
        f"lower values suggest irregular or overlapping/merged shapes). "
        f"Mean object intensity {mean_int:.2f} (0-1 scale)."
    )
    return summary, density


class NucleiSegDataset(Dataset):
    """
    PyTorch Dataset pairing grayscale nuclei images with their binary masks.

    Loads every (image, mask) pair listed in `image_paths` / `mask_paths`
    (must be the same length and in matching order), converts images to
    single-channel grayscale in [0, 1], and returns tensors shaped
    (1, H, W) for both image and mask so they feed directly into the U-Net.
    """

    def __init__(self, image_paths, mask_paths, size=(256, 256)):
        assert len(image_paths) == len(mask_paths), "image/mask list length mismatch"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        gray = to_grayscale_resized(load_image_rgb(self.image_paths[idx]), self.size)
        mask = load_mask_binary(self.mask_paths[idx], self.size)
        img_t = torch.from_numpy(gray).unsqueeze(0).float()          # (1, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()          # (1, H, W)
        return img_t, mask_t


def train_unet(model, train_loader, val_loader, loss_fn, epochs=15, lr=1e-3,
               device="cpu", verbose=True):
    """
    Train a U-Net-style model and track per-epoch train loss, val loss,
    val Dice, and val IoU. Returns a history dict with those four lists.

    Kept deliberately simple (single optimiser, no LR schedule) since the
    dataset is tiny (80 training images) and the assignment asks for a
    "modest number of epochs" rather than a fully tuned training regime.
    """
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimiser.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss_total, dice_total, iou_total, n_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = loss_fn(logits, masks)
                probs = torch.sigmoid(logits)
                val_loss_total += loss.item() * imgs.size(0)
                dice_total += dice_coefficient(probs, masks) * imgs.size(0)
                iou_total += iou_score(probs, masks) * imgs.size(0)
                n_batches += imgs.size(0)
        val_loss = val_loss_total / n_batches
        val_dice = dice_total / n_batches
        val_iou = iou_total / n_batches

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        if verbose:
            print(f"epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_dice={val_dice:.4f}  val_iou={val_iou:.4f}")

    return history


def evaluate_unet(model, data_loader, device="cpu"):
    """
    Run a trained model over a data loader and return the mean Dice and
    mean IoU across all batches, plus per-sample arrays (useful for
    finding which images the model does worst on).
    """
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for imgs, masks in data_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            probs = torch.sigmoid(model(imgs))
            for i in range(imgs.size(0)):
                dices.append(dice_coefficient(probs[i:i+1], masks[i:i+1]))
                ious.append(iou_score(probs[i:i+1], masks[i:i+1]))
    return float(np.mean(dices)), float(np.mean(ious)), np.array(dices), np.array(ious)



# 4. U-Net model (Task 3)

class DoubleConv(nn.Module):
    """Two 3x3 convolutions with batch norm and ReLU, the basic U-Net block.

    Verbatim from the module's Lab 4 (Lab4_CNN_unet_segmentation_SOLUTIONS.ipynb,
    "The U-Net Architecture" section), the provided U-Net implementation this
    assignment's Task 3 asks us to train.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    The small U-Net provided in the module's Lab 4, used as-is for Task 3
    (verbatim architecture, only the variable/class names are unchanged from
    the lab so it's traceable back to the source). 3 encoder stages, a
    bottleneck, 3 decoder stages with skip connections -- roughly 500K
    parameters with base=16, small enough to train on CPU in a few minutes.
    """

    def __init__(self, in_ch=1, out_ch=1, base=16):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        # Bottleneck
        self.bottleneck = DoubleConv(base * 4, base * 8)
        # Decoder
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)  # base*8 because of skip concat
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        # Output
        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder path (keep features for skip connections)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        # Decoder path with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))   # skip from e3
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))   # skip from e2
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))   # skip from e1
        return self.out_conv(d1)  # raw logits, shape (B, 1, H, W)


# Backward-compatible alias: earlier versions of this module used the name
# SmallUNet for a (non-provided) 4-level architecture. Everything in this
# project now uses the Lab 4-provided 3-level UNet above; this alias just
# avoids breaking any code/notebooks that still import SmallUNet by name.
SmallUNet = UNet


# 5. Losses & evaluation metrics (Task 3, extension loss ablation)

def dice_coefficient(pred_prob, target, eps=1e-6):
    """
    Soft/hard Dice coefficient between predicted probabilities/binary mask
    and target binary mask. Works batched: pred_prob, target shape (B,1,H,W).
    Returns the mean Dice over the batch as a python float.
    """
    pred = (pred_prob > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score(pred_prob, target, eps=1e-6):
    """Intersection-over-Union between predicted binary mask and target."""
    pred = (pred_prob > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union_area = ((pred + target) > 0).float().sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union_area + eps)
    return iou.mean().item()


class DiceLoss(nn.Module):
    """Soft Dice loss (1 - soft Dice), used alone or combined with BCE."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        intersection = (prob * target).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss (equal weighting), a common U-Net default."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, target):
        return self.bce(logits, target) + self.dice(logits, target)


# 6. Small helpers shared across tasks

def list_split(root, split):
    """Return sorted (image_paths, mask_paths) lists for a given split name."""
    img_dir = os.path.join(root, split, "images")
    mask_dir = os.path.join(root, split, "masks")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    return imgs, masks


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")
