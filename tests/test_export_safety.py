"""Safe checkpoint loading tests."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from training.export import print_model_info


def test_print_model_info_on_invalid_file(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    junk = tmp_path / "not_a_checkpoint.pt"
    junk.write_bytes(b"not-a-pytorch-checkpoint")
    print_model_info(junk)
    assert any(
        "Skipping pickle inspection" in rec.message or "Checkpoint size" in rec.message
        for rec in caplog.records
    )


def test_print_model_info_does_not_use_unsafe_pickle(tmp_path, monkeypatch, caplog):
    class Boom(pickle.Unpickler):
        def find_class(self, module, name):
            raise AssertionError("unsafe unpickler should not run")

    monkeypatch.setattr(pickle, "Unpickler", Boom)
    payload = tmp_path / "weights.pt"
    payload.write_bytes(b"\x80\x04N.")  # empty pickle protocol-4 None
    print_model_info(payload)
    assert "unsafe unpickler should not run" not in caplog.text


def test_export_module_uses_weights_only():
    source = Path("training/export.py").read_text()
    assert "weights_only=True" in source
    assert 'torch.load(checkpoint_path, map_location="cpu")' not in source.replace(
        'torch.load(checkpoint_path, map_location="cpu", weights_only=True)',
        "",
    )
