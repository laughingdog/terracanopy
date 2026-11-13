"""Generate a synthetic aerial forest image with exact ground truth.

Why synthetic: we know precisely which pixels are canopy and which trees are
stressed, so the analyzer's output can be checked against truth instead of
eyeballed. Output is deterministic (fixed seed).

Run (ephemeral env; does not touch the project's .venv):

    uv run --no-project --python 3.12 --with numpy --with opencv-python-headless \
        data_source/make_synthetic_forest.py

Writes next to this script:
    forest_synthetic.jpg          the image (JPEG, like a real camera/drone still)
    forest_synthetic_canopy.png   ground-truth canopy mask (255 = tree crown)
    forest_synthetic_truth.json   summary numbers + per-tree health labels
"""

import json
from pathlib import Path

import cv2
import numpy as np

SEED = 42
W, H = 960, 720
OUT = Path(__file__).parent

rng = np.random.default_rng(SEED)

# --- background: dry grass / soil, with low-frequency texture -----------------
def smooth_noise(scale: int, amp: float) -> np.ndarray:
    small = rng.normal(0, 1, (H // scale + 2, W // scale + 2)).astype(np.float32)
    return amp * cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


# One shared brightness field (patchy dry/damp ground) plus a faint per-channel tint,
# so the ground varies in brightness but stays brownish rather than rainbow-coloured.
bright = smooth_noise(40, 0.10)
base = np.empty((H, W, 3), np.float32)  # RGB, 0..255
base[..., 0] = 140 * (1 + bright) + smooth_noise(60, 2)  # tan/brown ground
base[..., 1] = 118 * (1 + bright) + smooth_noise(60, 2)
base[..., 2] = 82 * (1 + bright) + smooth_noise(60, 2)

# Non-canopy features the analyzer must NOT count as vegetation.
features = np.zeros((H, W), np.uint8)
# dirt road (diagonal band)
cv2.line(features, (-20, 640), (W + 20, 520), 255, 46)
# pond (dark blue-grey, an ellipse)
cv2.ellipse(features, (780, 150), (95, 60), 15, 0, 360, 128, -1)
road = features == 255
pond = features == 128
base[road] = np.array([176, 158, 126]) + rng.normal(0, 5, (road.sum(), 3))
base[pond] = np.array([52, 74, 88]) + rng.normal(0, 4, (pond.sum(), 3))

# --- trees: overlapping crowns, each healthy or stressed ----------------------
HEALTHY = [(38, 112, 40), (46, 128, 44), (30, 98, 38), (56, 138, 52)]  # RGB greens
STRESSED = [(122, 120, 48), (140, 104, 54), (112, 96, 44)]  # yellow/brown

canopy = np.zeros((H, W), np.uint8)
trees = []
placed = 0
while placed < 140:
    x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
    r = int(rng.integers(16, 44))
    # keep crowns off the road and pond
    probe = np.zeros((H, W), np.uint8)
    cv2.circle(probe, (x, y), r, 255, -1)
    if (probe.astype(bool) & (road | pond)).any():
        continue
    is_stressed = bool(rng.random() < 0.22)
    palette = STRESSED if is_stressed else HEALTHY
    color = np.array(palette[int(rng.integers(len(palette)))], np.float32)

    crown = probe.astype(bool)
    # soft shadow offset down-right (drawn first, darkens whatever is below)
    shadow = np.zeros((H, W), np.uint8)
    cv2.circle(shadow, (x + r // 3, y + r // 3), int(r * 1.05), 255, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 4).astype(np.float32) / 255.0
    base *= (1.0 - 0.35 * shadow)[..., None]

    # crown shading: bright centre, darker rim, plus leaf-scale texture
    yy, xx = np.ogrid[:H, :W]
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / r
    shade = np.clip(1.15 - 0.45 * d, 0.55, 1.2)
    leaf = rng.normal(1.0, 0.08, (H, W)).astype(np.float32)
    leaf = cv2.GaussianBlur(leaf, (0, 0), 1.5)
    px = color[None, None, :] * (shade * leaf)[..., None]
    base[crown] = px[crown]

    canopy[crown] = 255
    trees.append({"x": x, "y": y, "radius": r, "health": "stressed" if is_stressed else "healthy"})
    placed += 1

# Later crowns overwrite earlier ones, so derive the stressed truth from what is
# actually visible: a pixel is "stressed" only if the top-most crown there is.
stressed_mask = np.zeros((H, W), np.uint8)
order_canvas = np.zeros((H, W), np.int32)  # id of top-most tree per pixel, 0 = none
for i, t in enumerate(trees, start=1):
    tmp = np.zeros((H, W), np.uint8)
    cv2.circle(tmp, (t["x"], t["y"]), t["radius"], 255, -1)
    order_canvas[tmp.astype(bool)] = i
for i, t in enumerate(trees, start=1):
    if t["health"] == "stressed":
        stressed_mask[order_canvas == i] = 255

# --- global lighting gradient + sensor noise, then JPEG -----------------------
gx = np.linspace(0.92, 1.06, W, dtype=np.float32)[None, :, None]
gy = np.linspace(1.04, 0.94, H, dtype=np.float32)[:, None, None]
img = base * gx * gy + rng.normal(0, 3.0, base.shape).astype(np.float32)
img = np.clip(img, 0, 255).astype(np.uint8)

cv2.imwrite(str(OUT / "forest_synthetic.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 90])
cv2.imwrite(str(OUT / "forest_synthetic_canopy.png"), canopy)

total = H * W
truth = {
    "seed": SEED,
    "size": {"width": W, "height": H},
    "canopy_fraction": round(float((canopy > 0).sum() / total), 4),
    "stressed_fraction_of_canopy": round(
        float((stressed_mask > 0).sum() / max((canopy > 0).sum(), 1)), 4
    ),
    "n_trees": len(trees),
    "n_stressed_trees": sum(t["health"] == "stressed" for t in trees),
    "non_canopy_features": ["dirt road (diagonal)", "pond (upper right)"],
    "trees": trees,
}
(OUT / "forest_synthetic_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
print(json.dumps({k: v for k, v in truth.items() if k != "trees"}, indent=2))
