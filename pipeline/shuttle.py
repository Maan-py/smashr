"""Pembersihan hasil pelacakan kok."""
import numpy as np
import pandas as pd

RADIUS_PX = 24
MIN_STATIC_FRAMES = 12  # 0,4 detik pada 30 fps; kok yang dimainkan tidak diam selama ini

HIT_WINDOW = 3  # frame sebelum dan sesudah untuk membaca arah kok
HIT_MIN_STEP_PX = 12  # kok harus benar-benar bergerak di kedua sisi
HIT_MIN_TURN_DEG = 70
HIT_MIN_GAP = 15  # satu pukulan menghasilkan satu tanda


def detect_hits(track: pd.DataFrame) -> list[int]:
    """Perkirakan frame pukulan dari belokan tajam lintasan kok."""
    visible = (track["Visibility"] == 1).to_numpy()
    xy = track[["X", "Y"]].to_numpy(dtype=float)

    candidates = []
    for i in range(HIT_WINDOW, len(xy) - HIT_WINDOW):
        if not visible[i - HIT_WINDOW : i + HIT_WINDOW + 1].all():
            continue
        before = xy[i] - xy[i - HIT_WINDOW]
        after = xy[i + HIT_WINDOW] - xy[i]
        lengths = np.linalg.norm(before) * np.linalg.norm(after)
        if min(np.linalg.norm(before), np.linalg.norm(after)) < HIT_MIN_STEP_PX:
            continue
        turn = np.degrees(np.arccos(np.clip(before @ after / lengths, -1, 1)))
        if turn >= HIT_MIN_TURN_DEG:
            candidates.append((i, turn))

    hits: list[tuple[int, float]] = []
    for frame, turn in candidates:
        if hits and frame - hits[-1][0] < HIT_MIN_GAP:
            if turn > hits[-1][1]:
                hits[-1] = (frame, turn)
        else:
            hits.append((frame, turn))
    return [frame for frame, _ in hits]


def drop_static_points(track: pd.DataFrame) -> pd.DataFrame:
    """Tandai titik yang diam sebagai tidak terlihat.

    TrackNetV3 kadang macet di satu titik latar saat kehilangan kok, dan juga
    mengikuti kok yang tergeletak di lantai. Keduanya bukan kok yang sedang dimainkan.
    Butuh kolom Visibility, X, dan Y; DataFrame asli tidak diubah.
    """
    cleaned = track.copy()
    visible = cleaned.index[cleaned["Visibility"] == 1].to_numpy()
    points = cleaned.loc[visible, ["X", "Y"]].to_numpy(dtype=float)

    start = 0
    while start < len(points):
        end = start + 1
        while end < len(points) and np.hypot(*(points[end] - points[start])) < RADIUS_PX:
            end += 1
        if end - start >= MIN_STATIC_FRAMES:
            cleaned.loc[visible[start:end], "Visibility"] = 0
            start = end
        else:
            start += 1
    return cleaned
