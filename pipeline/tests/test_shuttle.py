import pandas as pd

from shuttle import MIN_STATIC_FRAMES, drop_static_points


def track(points):
    return pd.DataFrame(
        {"Visibility": [1] * len(points), "X": [p[0] for p in points], "Y": [p[1] for p in points]}
    )


def moving(n, start=0):
    return [(start + i * 40, 300) for i in range(n)]


def test_static_run_is_removed_and_moving_points_stay():
    static = [(608, 176)] * MIN_STATIC_FRAMES
    result = drop_static_points(track(moving(5) + static + moving(5, 800)))
    assert result["Visibility"].tolist() == [1] * 5 + [0] * MIN_STATIC_FRAMES + [1] * 5


def test_short_pause_is_kept():
    pause = [(608, 176)] * (MIN_STATIC_FRAMES - 1)
    result = drop_static_points(track(moving(5) + pause + moving(5, 800)))
    assert result["Visibility"].eq(1).all()


def test_slow_drift_inside_radius_counts_as_static():
    drift = [(600 + i, 200) for i in range(MIN_STATIC_FRAMES)]
    result = drop_static_points(track(drift))
    assert result["Visibility"].eq(0).all()


def test_input_is_not_modified():
    original = track([(608, 176)] * MIN_STATIC_FRAMES)
    drop_static_points(original)
    assert original["Visibility"].eq(1).all()
