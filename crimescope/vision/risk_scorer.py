import torch
import timm
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import polars as pl
import json
from torchvision import transforms
from crimescope.config import settings
from crimescope.utils.logger import logger

# ── Constants ─────────────────────────────────────────────────────

SCORES_PATH = settings.artifacts_dir / "vision" / "zone_risk_scores.json"
SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)

# Image preprocessing for EfficientNet
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ── Model ─────────────────────────────────────────────────────────

def load_model() -> torch.nn.Module:
    """
    Load pretrained EfficientNet-B0 from TIMM.
    We use it as a feature extractor — not for classification.
    The model was trained on ImageNet so it understands
    visual concepts like roads, buildings, vegetation, lighting.
    """

    logger.info("Loading EfficientNet-B0 from TIMM...")
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=0,      # remove classifier head → pure feature extractor
    )
    model.eval()
    logger.success("Model loaded ✓")
    return model


# ── Visual Risk Features ──────────────────────────────────────────

def compute_visual_features(image_path: Path) -> dict:
    """
    Extract interpretable visual risk signals from a map tile image.
    These are criminology-informed features:

    - darkness_score: darker areas = less lighting = higher risk
    - edge_density: more edges = more visual complexity = urban density
    - color_entropy: more color variation = more urban features
    - green_ratio: more green = parks/vegetation (lower risk)
    - gray_ratio: more gray = concrete/roads (urban core)
    """

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return {}

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Darkness score (0=bright, 1=dark) ────────────────────────
    mean_brightness = img_gray.mean() / 255.0
    darkness_score = round(1.0 - mean_brightness, 4)

    # ── Edge density — Canny edge detection ──────────────────────
    edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)
    edge_density = round(edges.mean() / 255.0, 4)

    # ── Color entropy ─────────────────────────────────────────────
    hist = cv2.calcHist([img_rgb], [0, 1, 2], None,
                        [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    hist = hist[hist > 0]
    prob = hist / hist.sum()
    entropy = round(float(-np.sum(prob * np.log2(prob + 1e-9))), 4)

    # ── Color ratios ──────────────────────────────────────────────
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]

    # Green pixels: G dominates R and B (vegetation)
    green_mask = (g > r + 20) & (g > b + 20)
    green_ratio = round(float(green_mask.sum() / total_pixels), 4)

    # Gray pixels: R≈G≈B (roads, concrete)
    diff_rg = np.abs(r.astype(int) - g.astype(int))
    diff_rb = np.abs(r.astype(int) - b.astype(int))
    gray_mask = (diff_rg < 20) & (diff_rb < 20)
    gray_ratio = round(float(gray_mask.sum() / total_pixels), 4)

    return {
        "darkness_score": darkness_score,
        "edge_density": edge_density,
        "color_entropy": entropy,
        "green_ratio": green_ratio,
        "gray_ratio": gray_ratio,
    }


def compute_deep_features(
    image_path: Path,
    model: torch.nn.Module,
) -> np.ndarray:
    """
    Extract 1280-dim feature vector from EfficientNet.
    These deep features capture complex visual patterns
    that our hand-crafted features might miss.
    """

    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        features = model(tensor)

    return features.squeeze().numpy()


# ── Risk Score ────────────────────────────────────────────────────

def compute_risk_score(visual_features: dict) -> float:
    """
    Combine visual features into a single risk score (0-100).

    Weights based on criminology research:
    - Darkness → highest weight (poor lighting = more crime)
    - Edge density → urban complexity
    - Low green → less natural surveillance
    - High gray → dense urban core
    - Entropy → chaotic environments
    """

    if not visual_features:
        return 50.0  # neutral default

    weights = {
        "darkness_score": 0.35,
        "edge_density":   0.25,
        "color_entropy":  0.15,
        "green_ratio":   -0.15,  # negative — green REDUCES risk
        "gray_ratio":     0.10,
    }

    score = sum(
        visual_features.get(feat, 0) * weight
        for feat, weight in weights.items()
    )

    # Normalize to 0-100
    score_normalized = round(min(max(score * 100, 0), 100), 2)
    return score_normalized


# ── Full Scoring Pipeline ─────────────────────────────────────────

def score_all_zones(image_paths: dict[int, Path]) -> dict[int, dict]:
    """
    Score all zones and save results.
    Returns dict of {zone_id: {risk_score, visual_features}}
    """

    logger.info(f"Scoring {len(image_paths)} zone images...")
    model = load_model()
    all_scores = {}

    for zone_id, img_path in image_paths.items():
        visual_feats = compute_visual_features(img_path)
        risk_score = compute_risk_score(visual_feats)

        all_scores[zone_id] = {
            "zone_id": zone_id,
            "risk_score": risk_score,
            **visual_feats,
        }

        logger.debug(
            f"Zone {zone_id} → risk score: {risk_score:.1f} "
            f"(dark={visual_feats.get('darkness_score', 0):.2f}, "
            f"edges={visual_feats.get('edge_density', 0):.2f})"
        )

    # Save scores to JSON
    with open(SCORES_PATH, "w") as f:
        json.dump(all_scores, f, indent=2)

    logger.success(f"Risk scores saved → {SCORES_PATH}")
    return all_scores


def load_risk_scores() -> pl.DataFrame:
    """
    Load saved risk scores as a Polars DataFrame.
    Used to merge vision features back into the ML pipeline.
    """

    with open(SCORES_PATH) as f:
        data = json.load(f)

    rows = list(data.values())
    df = pl.DataFrame(rows).with_columns(
        pl.col("zone_id").cast(pl.Int32)
    )

    logger.info(f"Loaded risk scores for {df.shape[0]} zones")
    return df


# ── Runner ────────────────────────────────────────────────────────

def run_vision_pipeline(zone_ids: list[int]) -> pl.DataFrame:
    """Full vision pipeline — fetch + score + return DataFrame."""

    from crimescope.vision.street_fetcher import fetch_all_zones

    logger.info("=" * 50)
    logger.info("Starting Vision Pipeline")
    logger.info("=" * 50)

    # Fetch map tile images
    image_paths = fetch_all_zones(zone_ids)

    # Score all images
    scores = score_all_zones(image_paths)

    # Return as DataFrame
    return load_risk_scores()