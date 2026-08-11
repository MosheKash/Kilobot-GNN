import os
import tempfile
import torch
import pytest

from images import formation_paths, build_image_pool
from trainer import _sorted_png_position


def _fake_preprocess(path):
    return torch.zeros(1, 1, 4, 4)


def test_formation_paths_filters_by_pattern_and_sorts():
    with tempfile.TemporaryDirectory() as d:
        for name in ["b.png", "a.png", "c.txt"]:
            open(os.path.join(d, name), "w").close()
        paths = formation_paths(d, pattern = ".png")
        names = [os.path.basename(p) for p in paths]
        assert names == ["a.png", "b.png"]


def test_formation_paths_respects_limit():
    with tempfile.TemporaryDirectory() as d:
        for name in ["a.png", "b.png", "c.png"]:
            open(os.path.join(d, name), "w").close()
        paths = formation_paths(d, pattern = ".png", limit = 2)
        assert len(paths) == 2


def test_sorted_png_position_matches_unity_position_index():
    # With a CONTIGUOUS %06d folder, sorted position == numeric name, which is
    # what the old implementation (int(stem)) got right by accident.
    with tempfile.TemporaryDirectory() as d:
        for i in range(5):
            open(os.path.join(d, "%06d.png" % i), "w").close()
        assert _sorted_png_position(d, "000003.png") == 3


def test_sorted_png_position_handles_non_contiguous_folder():
    # The regressive case that broke watch_actor.sh: a subsampled folder whose
    # numeric names are NOT their sorted position. val_formations is one such:
    # position 54 there is not 000054.png. Python MUST send Unity the sorted
    # position (what ImageLibrary.files[] indexes), never the numeric stem.
    with tempfile.TemporaryDirectory() as d:
        for name in ["000054.png", "000086.png", "004025.png", "000157.png"]:
            open(os.path.join(d, name), "w").close()
        assert _sorted_png_position(d, "000054.png") == 0
        assert _sorted_png_position(d, "000157.png") == 2  # not 157
        assert _sorted_png_position(d, "004025.png") == 3  # not 4025


def test_sorted_png_position_missing_name_returns_negative():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.png"), "w").close()
        assert _sorted_png_position(d, "missing.png") == -1


def test_build_image_pool_default_device_is_whatever_preprocess_returns():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.png"), "w").close()
        pool = build_image_pool(d, _fake_preprocess, limit = 1)
        assert pool[0].device.type == "cpu"


def test_build_image_pool_explicit_cpu_device():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.png"), "w").close()
        pool = build_image_pool(d, _fake_preprocess, limit = 1, device = "cpu")
        assert pool[0].device.type == "cpu"


class _RecordingTensor:
    def __init__(self):
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


def test_build_image_pool_calls_to_with_the_requested_device():
    def preprocess_recording(path):
        return _RecordingTensor()
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.png"), "w").close()
        pool = build_image_pool(d, preprocess_recording, limit = 1, device = "cuda")
        assert pool[0].to_calls == ["cuda"]


def test_build_image_pool_default_does_not_call_to_at_all():
    def preprocess_recording(path):
        return _RecordingTensor()
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.png"), "w").close()
        pool = build_image_pool(d, preprocess_recording, limit = 1)
        assert pool[0].to_calls == []


def test_build_image_pool_preserves_tensor_content():
    def preprocess_marked(path):
        return torch.full((1, 1, 4, 4), 7.0)
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.png"), "w").close()
        pool = build_image_pool(d, preprocess_marked, limit = 1, device = "cpu")
        assert torch.allclose(pool[0], torch.full((1, 1, 4, 4), 7.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_build_image_pool_moves_to_cuda_when_requested():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.png"), "w").close()
        pool = build_image_pool(d, _fake_preprocess, limit = 1, device = "cuda")
        assert pool[0].device.type == "cuda"


def _preprocess_real_size(path):
    from encoder import INPUT_SIZE
    return torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_encoder_on_cuda_can_encode_a_pool_built_for_cuda():
    from encoder import ConvAutoencoder, LatentEncoder
    autoencoder = ConvAutoencoder(latent_dim = 8).to("cuda")
    encoder = LatentEncoder(autoencoder).to("cuda")
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.png"), "w").close()
        pool = build_image_pool(d, _preprocess_real_size, limit = 1, device = "cuda")
        with torch.no_grad():
            z = encoder(pool[0])
        assert z.device.type == "cuda"
