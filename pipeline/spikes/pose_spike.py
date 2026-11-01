"""Fase 0: ukur kecepatan pose di CPU dan seberapa sering kaki pemain terlihat.

Pakai: python pose_spike.py VIDEO OUT_DIR [--samples 240]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Body

L_ANKLE, R_ANKLE = 15, 16
MIN_SCORE = 0.3
MIN_BODY_PX = 90  # abaikan orang kecil di kejauhan
BOTTOM_MARGIN_PX = 6


def body_height(kpts, scores):
    ok = scores > MIN_SCORE
    return float(np.ptp(kpts[ok, 1])) if ok.sum() >= 5 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--samples", type=int, default=240)
    ap.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = Body(mode=args.mode, backend="onnxruntime", device="cpu")
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idxs = np.linspace(0, total - 1, args.samples, dtype=int)

    infer_s, rows = [], []
    for n, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        t0 = time.perf_counter()
        kpts, scores = model(frame)
        infer_s.append(time.perf_counter() - t0)

        people = []
        for k, s in zip(kpts, scores):
            h = body_height(k, s)
            if h < MIN_BODY_PX:
                continue
            ankles_seen = sum(
                s[a] > MIN_SCORE and k[a, 1] < height - BOTTOM_MARGIN_PX
                for a in (L_ANKLE, R_ANKLE)
            )
            people.append({"h": h, "bottom": float(k[s > MIN_SCORE, 1].max()), "ankles": int(ankles_seen)})
        people.sort(key=lambda p: p["bottom"], reverse=True)
        rows.append({"frame": int(idx), "people": people})

        if n % 60 == 0:
            vis = frame.copy()
            for k, s in zip(kpts, scores):
                for (x, y), sc in zip(k, s):
                    if sc > MIN_SCORE:
                        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)
            cv2.imwrite(str(out / f"pose_{idx}.jpg"), vis)

    near = [r["people"][0] for r in rows if r["people"]]
    summary = {
        "frames_sampled": len(rows),
        "sec_per_frame_mean": round(float(np.mean(infer_s[1:])), 3),
        "frames_with_person": len(near),
        "frames_with_2plus_people": sum(len(r["people"]) >= 2 for r in rows),
        "near_player_both_ankles_visible_pct": round(100 * np.mean([p["ankles"] == 2 for p in near]), 1),
        "near_player_any_ankle_visible_pct": round(100 * np.mean([p["ankles"] >= 1 for p in near]), 1),
    }
    (out / "pose_rows.json").write_text(json.dumps(rows))
    (out / "pose_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
