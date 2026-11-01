"""Pembersihan hasil pelacakan kok."""
import numpy as np
import pandas as pd

RADIUS_PX = 24
MIN_STATIC_FRAMES = 12  # 0,4 detik pada 30 fps; kok yang dimainkan tidak diam selama ini


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
