import json

import cv2
import numpy as np
import pytest

from terracanopy import Frame, Pipeline, available_indices, compute_index
from terracanopy.analyzers import VegetationAnalyzer
from terracanopy.cli import main
from terracanopy.sinks import JsonlSink
from terracanopy.sources import ImageSource, VideoSource


def synthetic_rgb(h=64, w=64):
    """Left half green 'canopy', right half brown 'soil'."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 2] = (30, 160, 40)
    img[:, w // 2 :] = (130, 100, 70)
    return img


def test_frame_validation():
    with pytest.raises(ValueError):
        Frame(np.zeros((3, 4)), ("red", "green", "blue"))
    with pytest.raises(ValueError):
        Frame(np.zeros((2, 4, 4)), ("red", "green", "blue"))


def test_available_indices_depends_on_bands():
    rgb = Frame.from_rgb_uint8(synthetic_rgb())
    assert set(available_indices(rgb)) == {"exg", "vari"}
    ms = Frame.from_bands({k: np.full((4, 4), 0.3, np.float32) for k in ("red", "green", "nir")})
    assert set(available_indices(ms)) == {"ndvi", "gndvi", "ndwi"}


def test_ndvi_values():
    red = np.full((4, 4), 0.1, np.float32)
    nir = np.full((4, 4), 0.5, np.float32)
    f = Frame.from_bands({"red": red, "nir": nir})
    np.testing.assert_allclose(compute_index(f, "ndvi"), (0.5 - 0.1) / (0.5 + 0.1), rtol=1e-5)


def test_missing_band_raises():
    f = Frame.from_rgb_uint8(synthetic_rgb())
    with pytest.raises(ValueError):
        compute_index(f, "ndvi")


def test_zero_denominator_is_safe():
    z = np.zeros((4, 4), np.float32)
    f = Frame.from_bands({"red": z, "nir": z})
    out = compute_index(f, "ndvi")
    assert np.isfinite(out).all()


def test_vegetation_analyzer_separates_canopy_from_soil_rgb():
    f = Frame.from_rgb_uint8(synthetic_rgb())
    r = VegetationAnalyzer().analyze(f)
    mask = r.layers["mask"]
    assert mask[:, :32].all() and not mask[:, 32:].any()
    assert r.metrics["canopy_fraction"] == pytest.approx(0.5)
    assert r.info["index"] in {"exg", "vari"}


def test_vegetation_analyzer_prefers_ndvi_with_nir():
    h = w = 16
    red = np.full((h, w), 0.4, np.float32)
    nir = np.full((h, w), 0.4, np.float32)
    red[:, : w // 2] = 0.05  # healthy: low red, high NIR
    nir[:, : w // 2] = 0.6
    f = Frame.from_bands({"red": red, "green": red.copy(), "blue": red.copy(), "nir": nir})
    r = VegetationAnalyzer().analyze(f)
    assert r.info["index"] == "ndvi"
    assert r.layers["mask"][:, : w // 2].all()
    assert not r.layers["mask"][:, w // 2 :].any()


def test_min_blob_removes_speckle():
    img = np.full((64, 64, 3), (130, 100, 70), np.uint8)
    img[10, 10] = (30, 160, 40)  # single-pixel speck
    img[30:50, 30:50] = (30, 160, 40)  # real blob
    f = Frame.from_rgb_uint8(img)
    r = VegetationAnalyzer(index="exg", threshold=0.1, min_blob_px=10).analyze(f)
    assert not r.layers["mask"][10, 10]
    assert r.layers["mask"][40, 40]


def test_image_source_and_pipeline(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"img{i}.png"), cv2.cvtColor(synthetic_rgb(), cv2.COLOR_RGB2BGR))
    out = tmp_path / "res.jsonl"
    pipe = Pipeline(ImageSource(tmp_path), [VegetationAnalyzer()], [JsonlSink(out)])
    results = pipe.run()
    assert len(results) == 3
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 3
    assert "latency_ms" in lines[0]["metrics"]


def test_video_source_stride(tmp_path):
    path = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64))
    for _ in range(10):
        vw.write(cv2.cvtColor(synthetic_rgb(), cv2.COLOR_RGB2BGR))
    vw.release()
    frames = list(VideoSource(path, stride=3))
    assert [f.index for f in frames] == [0, 3, 6, 9]


def test_cli_end_to_end(tmp_path, capsys):
    img = tmp_path / "tree.png"
    cv2.imwrite(str(img), cv2.cvtColor(synthetic_rgb(), cv2.COLOR_RGB2BGR))
    out = tmp_path / "out"
    assert main(["analyze", str(img), "--out", str(out)]) == 0
    assert (out / "results.jsonl").exists()
    assert len(list((out / "overlays").glob("*.png"))) == 1
    assert "canopy=" in capsys.readouterr().out
