import numpy as np

from robot_soccer_vision.difficulty import (
    DifficultyRecord,
    _assign_composite_difficulty,
    sample_difficult_frames,
)


def record(index: int, confidence: float, blur: float, brightness: float) -> DifficultyRecord:
    return DifficultyRecord(
        frame_index=index,
        teacher_x=100.0,
        teacher_y=100.0,
        teacher_radius=12.0,
        teacher_confidence=confidence,
        teacher_votes=3,
        student_teacher_distance=float(index),
        temporal_surprise=float(index),
        blur_score=blur,
        median_brightness=brightness,
        shadow_fraction=max(0.0, (100 - brightness) / 100),
    )


def test_difficult_conditions_receive_higher_score() -> None:
    easy = record(0, confidence=0.95, blur=500, brightness=120)
    difficult = record(1, confidence=0.25, blur=5, brightness=15)

    _assign_composite_difficulty([easy, difficult], radius=12)

    assert difficult.difficulty > easy.difficulty


def test_random_sample_comes_only_from_difficult_pool_and_is_reproducible() -> None:
    records = [DifficultyRecord(frame_index=i, difficulty=i / 100) for i in range(100)]

    first = sample_difficult_frames(records, count=10, pool_fraction=0.2, seed=42)
    second = sample_difficult_frames(records, count=10, pool_fraction=0.2, seed=42)

    assert [item.frame_index for item in first] == [item.frame_index for item in second]
    assert all(item.frame_index >= 80 for item in first)
    assert len({item.frame_index for item in first}) == 10


def test_completed_frames_are_excluded_from_new_sample() -> None:
    records = [DifficultyRecord(frame_index=i, difficulty=float(i)) for i in range(20)]

    selected = sample_difficult_frames(
        records, count=5, pool_fraction=1.0, seed=3, excluded_indices={2, 4, 6, 8}
    )

    assert not ({item.frame_index for item in selected} & {2, 4, 6, 8})
