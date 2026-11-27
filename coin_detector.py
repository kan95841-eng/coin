from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

DENOMINATIONS = (50, 20, 10, 5, 1)

# Official NTD coin diameters in mm
COIN_DIAMETERS_MM: dict[int, float] = {
    50: 28.0,
    20: 26.85,
    10: 26.0,
    5: 22.0,
    1: 20.0,
}

# BGR/RGB annotation colors for each denomination
COIN_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    50: (255, 193, 7),    # Gold
    20: (156, 39, 176),   # Purple
    10: (33, 150, 243),   # Blue
    5: (0, 188, 212),     # Cyan
    1: (255, 87, 34),     # Copper/Orange
}


@dataclass
class DetectedCoin:
    x: int
    y: int
    r: int
    denomination: int


@dataclass
class DetectionResult:
    counts: dict[int, int]
    total_amount: int
    annotated_image: np.ndarray
    coins: list[DetectedCoin]


def calculate_total(counts: dict[int, int]) -> int:
    """Calculate total NTD amount from a mapping of {denomination: count}."""
    return sum(int(denom) * int(qty) for denom, qty in counts.items() if int(qty) > 0)


def _sample_ring_hsv(
    hsv_img: np.ndarray, cx: int, cy: int, r_inner: float, r_outer: float
) -> tuple[float, float, float]:
    """Return median (H, S, V) inside an annular region around (cx, cy)."""
    h, w = hsv_img.shape[:2]
    y1 = max(0, int(cy - r_outer))
    y2 = min(h, int(cy + r_outer) + 1)
    x1 = max(0, int(cx - r_outer))
    x2 = min(w, int(cx + r_outer) + 1)

    yy, xx = np.ogrid[y1:y2, x1:x2]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (dist_sq >= r_inner ** 2) & (dist_sq <= r_outer ** 2)

    pixels = hsv_img[y1:y2, x1:x2][mask]
    if len(pixels) == 0:
        return 0.0, 0.0, 0.0
    med = np.median(pixels, axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def _classify_coin(
    hsv_img: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    max_r: int,
) -> int:
    """Classify a single detected circle into NTD denomination (1, 5, 10, 20, 50)."""
    h_in, s_in, _ = _sample_ring_hsv(hsv_img, cx, cy, 0, r * 0.50)
    h_out, s_out, _ = _sample_ring_hsv(hsv_img, cx, cy, r * 0.58, r * 0.88)
    rel_size = r / max_r if max_r > 0 else 1.0

    # 20 NTD: bi-metallic (gold/yellowish center, silver/low-sat outer ring)
    if s_in >= 45 and s_out <= 45 and (s_in - s_out) >= 20 and 15 <= h_in <= 38:
        return 20

    # 50 NTD: brass/gold overall (yellow-gold hue, moderate-high saturation)
    if s_in >= 45 and s_out >= 35 and 16 <= h_in <= 38 and 16 <= h_out <= 38:
        return 50

    # 1 NTD: copper/bronze (reddish-orange hue, usually < 18 or > 165 in OpenCV H[0..179])
    if (s_in >= 35 or s_out >= 35) and (h_in < 16 or h_in > 165 or h_out < 16 or h_out > 165):
        return 1

    # Silver coins (5 NTD vs 10 NTD) — disambiguate by relative radius
    # 10 NTD (26mm) vs 5 NTD (22mm): ratio ~ 0.846
    if rel_size >= 0.86:
        return 10
    if rel_size >= 0.72:
        return 5
    return 1


def detect_coins(
    image_rgb: np.ndarray,
    dp: float = 1.2,
    param1: int = 60,
    param2: int = 38,
    min_radius: int = 18,
    max_radius: int = 140,
) -> DetectionResult:
    """Detect and classify NTD coins in an RGB image using offline OpenCV."""
    if image_rgb is None or image_rgb.size == 0:
        empty_counts = {d: 0 for d in DENOMINATIONS}
        return DetectionResult(
            counts=empty_counts,
            total_amount=0,
            annotated_image=np.zeros((100, 100, 3), dtype=np.uint8),
            coins=[],
        )

    annotated = image_rgb.copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    hsv_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    min_dist = max(20, int(min_radius * 1.8))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    counts: dict[int, int] = {d: 0 for d in DENOMINATIONS}
    detected_coins: list[DetectedCoin] = []

    if circles is not None:
        raw_circles = np.round(circles[0, :]).astype(int)
        max_r = int(np.max(raw_circles[:, 2]))

        for x, y, r in raw_circles:
            denom = _classify_coin(hsv_img, int(x), int(y), int(r), max_r)
            counts[denom] += 1
            detected_coins.append(DetectedCoin(x=int(x), y=int(y), r=int(r), denomination=denom))

            color = COIN_COLORS_RGB.get(denom, (0, 255, 0))
            cv2.circle(annotated, (int(x), int(y)), int(r), color, 3)
            label = f"{denom}"
            cv2.putText(
                annotated,
                label,
                (int(x) - 18, int(y) + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

    return DetectionResult(
        counts=counts,
        total_amount=calculate_total(counts),
        annotated_image=annotated,
        coins=detected_coins,
    )
