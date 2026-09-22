from hashlib import sha256
import json

from robot_soccer_vision.provenance import (
    artifact_records,
    file_record,
    json_safe,
    write_manifest,
)


def test_file_record_contains_reproducible_absolute_path_and_hash(tmp_path) -> None:
    source = tmp_path / "sample.bin"
    source.write_bytes(b"robot-soccer")

    record = file_record(source, "test_video")

    assert record["path"] == str(source.resolve())
    assert record["filename"] == "sample.bin"
    assert record["size_bytes"] == len(b"robot-soccer")
    assert record["sha256"] == sha256(b"robot-soccer").hexdigest()
    assert record["role"] == "test_video"


def test_manifest_hashes_artifacts_and_uses_strict_json(tmp_path) -> None:
    artifact = tmp_path / "results.csv"
    artifact.write_text("x,y\n1,2\n")
    manifest = tmp_path / "manifest.json"

    write_manifest(
        manifest,
        {
            "metric": float("inf"),
            "artifacts": artifact_records([artifact]),
        },
    )

    payload = json.loads(manifest.read_text())
    assert payload["metric"] is None
    assert payload["artifacts"][0]["sha256"] == sha256(artifact.read_bytes()).hexdigest()


def test_json_safe_converts_numpy_and_non_finite_values() -> None:
    import numpy as np

    assert json_safe({"value": np.float32(2.5), "bad": float("nan")}) == {
        "value": 2.5,
        "bad": None,
    }
