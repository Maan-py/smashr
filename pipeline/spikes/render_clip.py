"""Prototipe SMR-6a: klip story 1080x1920 dengan animasi kok dan pemain.

Pakai: python render_clip.py VIDEO BALL_CSV OUT_MP4
Batas lapangan di bawah khusus untuk rekaman uji, belum memakai kalibrasi.
"""
import argparse
import subprocess
import time

import cv2
import numpy as np
import pandas as pd
from rtmlib import Body

from shuttle import detect_hits, drop_static_points

W, H = 1080, 1920
VID_H = 608
VID_Y = (H - VID_H) // 2
SCALE = W / 1280
POSE_EVERY = 6
TRAIL = 14
MIN_SCORE = 0.3
COURT_X = (150, 1150)
COURT_MIN_FOOT_Y = 610
BG = (28, 24, 21)
SHUTTLE = (90, 240, 255)  # BGR
NEAR = (138, 195, 76)
FAR = (235, 160, 70)
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]


def players_in_frame(kpts, scores):
    """Pilih dua orang di lapangan: indeks 0 pemain dekat, indeks 1 pemain jauh."""
    found = []
    for k, s in zip(kpts, scores):
        ok = s > MIN_SCORE
        if ok.sum() < 8:
            continue
        foot_y, mid_x = k[ok, 1].max(), np.median(k[ok, 0])
        if foot_y >= COURT_MIN_FOOT_Y and COURT_X[0] < mid_x < COURT_X[1]:
            found.append((foot_y, k, s))
    found.sort(key=lambda p: p[0], reverse=True)
    return [(k, s) for _, k, s in found[:2]]


def track_players(video, fps_note=True):
    model = Body(mode="lightweight", backend="onnxruntime", device="cpu")
    cap = cv2.VideoCapture(video)
    samples, i = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % POSE_EVERY == 0:
            samples[i] = players_in_frame(*model(frame))
        i += 1
    cap.release()
    return samples


def player_at(samples, frame, role):
    """Interpolasi pose antar-sampel agar gerakan tetap halus."""
    a = frame - frame % POSE_EVERY
    b = a + POSE_EVERY
    sa, sb = samples.get(a, []), samples.get(b, [])
    if len(sa) <= role:
        return None
    k, s = sa[role]
    if len(sb) > role:
        t = (frame - a) / POSE_EVERY
        return k * (1 - t) + sb[role][0] * t, s
    return k, s


def to_canvas(x, y):
    return int(x * SCALE), int(y * SCALE) + VID_Y


def draw_player(canvas, player, color):
    k, s = player
    for a, b in BONES:
        if s[a] > MIN_SCORE and s[b] > MIN_SCORE:
            cv2.line(canvas, to_canvas(*k[a]), to_canvas(*k[b]), color, 3, cv2.LINE_AA)
    for (x, y), sc in zip(k, s):
        if sc > MIN_SCORE:
            cv2.circle(canvas, to_canvas(x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_trail(canvas, track, frame):
    visible = track["Visibility"].to_numpy() == 1
    xy = track[["X", "Y"]].to_numpy()
    last = None
    for i in range(max(0, frame - TRAIL), frame + 1):
        if i >= len(xy) or not visible[i]:
            last = None
            continue
        p = to_canvas(*xy[i])
        if last is not None and np.hypot(p[0] - last[0], p[1] - last[1]) < 220:
            weight = 1 + int(6 * (i - frame + TRAIL) / TRAIL)
            cv2.line(canvas, last, p, SHUTTLE, weight, cv2.LINE_AA)
        last = p
    if frame < len(xy) and visible[frame]:
        cv2.circle(canvas, to_canvas(*xy[frame]), 9, (255, 255, 255), 2, cv2.LINE_AA)


def text(canvas, value, size, y, color=(240, 240, 240), thick=2):
    font = cv2.FONT_HERSHEY_DUPLEX
    (w, _), _ = cv2.getTextSize(value, font, size, thick)
    cv2.putText(canvas, value, ((W - w) // 2, y), font, size, color, thick, cv2.LINE_AA)


def render(video, track, hits, samples, out):
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE,
    )
    cap = cv2.VideoCapture(video)
    frame_no, hit_set = 0, set(hits)
    count, flash = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = np.full((H, W, 3), BG, np.uint8)
        canvas[VID_Y:VID_Y + VID_H] = cv2.resize(frame, (W, VID_H), interpolation=cv2.INTER_AREA)

        for role, color in ((1, FAR), (0, NEAR)):
            player = player_at(samples, frame_no, role)
            if player is not None:
                draw_player(canvas, player, color)
        draw_trail(canvas, track, frame_no)

        if frame_no in hit_set:
            count += 1
            flash = 10
        if flash:
            x, y = track.loc[min(frame_no, len(track) - 1), ["X", "Y"]]
            cv2.circle(canvas, to_canvas(x, y), 12 + (10 - flash) * 4, (255, 255, 255), 3, cv2.LINE_AA)
            flash -= 1

        text(canvas, "SMASHR", 1.5, 190, (138, 195, 76), 3)
        text(canvas, str(count), 7, 480, (255, 255, 255), 12)
        text(canvas, "pukulan dalam klip ini", 1.5, 560)
        text(canvas, "Prototipe", 1.1, 1500, (170, 170, 170))
        ff.stdin.write(canvas.tobytes())
        frame_no += 1
    cap.release()
    ff.stdin.close()
    ff.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("ball_csv")
    ap.add_argument("out")
    args = ap.parse_args()

    track = drop_static_points(pd.read_csv(args.ball_csv))
    hits = detect_hits(track)

    t0 = time.perf_counter()
    samples = track_players(args.video)
    t1 = time.perf_counter()
    render(args.video, track, hits, samples, args.out)
    t2 = time.perf_counter()

    two = sum(len(p) == 2 for p in samples.values())
    print(f"hits={len(hits)} frames={hits}")
    print(f"pose {t1 - t0:.0f}s, render {t2 - t1:.0f}s, pose frames with 2 players {two}/{len(samples)}")


if __name__ == "__main__":
    main()
