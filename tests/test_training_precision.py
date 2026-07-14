import pytest

from medvlm_grpo import precision


def test_auto_precision_prefers_bf16_when_supported(monkeypatch):
    monkeypatch.setattr(precision.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(precision.torch.cuda, "is_bf16_supported", lambda: True)

    assert precision.resolve_precision("auto") == "bf16"
    assert precision.precision_kwargs("bf16") == {"bf16": True, "fp16": False}


def test_auto_precision_falls_back_to_fp16(monkeypatch):
    monkeypatch.setattr(precision.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(precision.torch.cuda, "is_bf16_supported", lambda: False)

    assert precision.resolve_precision("auto") == "fp16"
    assert precision.precision_kwargs("fp16") == {"bf16": False, "fp16": True}


def test_explicit_unsupported_bf16_fails_early(monkeypatch):
    monkeypatch.setattr(precision.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(precision.torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(RuntimeError, match="does not support BF16"):
        precision.resolve_precision("bf16")
