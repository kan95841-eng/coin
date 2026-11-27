import cv2
import numpy as np

from coin_detector import calculate_total, detect_coins


def test_calculate_total_basic():
    counts = {50: 2, 20: 1, 10: 3, 5: 1, 1: 4}
    assert calculate_total(counts) == 100 + 20 + 30 + 5 + 4


def test_calculate_total_empty_and_zeros():
    assert calculate_total({}) == 0
    assert calculate_total({50: 0, 10: 0, 1: 0}) == 0


def test_detect_coins_empty_image():
    blank = np.full((200, 200, 3), 240, dtype=np.uint8)
    result = detect_coins(blank)
    assert result.total_amount == 0
    assert len(result.coins) == 0


def test_detect_coins_synthetic_all_denominations():
    # Create a clean light-gray background
    img = np.full((420, 720, 3), 235, dtype=np.uint8)

    # Draw 5 synthetic coins with realistic relative radii & RGB colors:
    # 50 NTD: r=56, brass/gold RGB ~ (212, 175, 55)
    cv2.circle(img, (100, 210), 56, (212, 175, 55), -1)

    # 20 NTD: r=53, outer silver RGB ~ (185, 185, 185), inner gold RGB ~ (212, 175, 55)
    cv2.circle(img, (240, 210), 53, (185, 185, 185), -1)
    cv2.circle(img, (240, 210), 30, (212, 175, 55), -1)

    # 10 NTD: r=52, silver RGB ~ (190, 190, 190)
    cv2.circle(img, (380, 210), 52, (190, 190, 190), -1)

    # 5 NTD: r=44, silver RGB ~ (190, 190, 190)
    cv2.circle(img, (510, 210), 44, (190, 190, 190), -1)

    # 1 NTD: r=40, copper RGB ~ (184, 115, 51)
    cv2.circle(img, (625, 210), 40, (184, 115, 51), -1)

    result = detect_coins(img, param2=30, min_radius=30, max_radius=70)

    assert result.counts[50] == 1
    assert result.counts[20] == 1
    assert result.counts[10] == 1
    assert result.counts[5] == 1
    assert result.counts[1] == 1
    assert result.total_amount == 86
    assert len(result.coins) == 5
