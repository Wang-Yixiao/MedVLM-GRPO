import sys
import types

from medvlm_grpo.tracking import make_swanlab_callback


def test_swanlab_is_not_imported_when_disabled(monkeypatch):
    monkeypatch.setitem(sys.modules, "swanlab", None)
    assert make_swanlab_callback(enabled=False, project="test") is None


def test_swanlab_callback_receives_run_metadata(monkeypatch):
    class FakeCallback:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    integration = types.ModuleType("swanlab.integration")
    transformers = types.ModuleType("swanlab.integration.transformers")
    transformers.SwanLabCallback = FakeCallback
    monkeypatch.setitem(sys.modules, "swanlab", types.ModuleType("swanlab"))
    monkeypatch.setitem(sys.modules, "swanlab.integration", integration)
    monkeypatch.setitem(sys.modules, "swanlab.integration.transformers", transformers)

    callback = make_swanlab_callback(
        enabled=True,
        project="medical-vqa",
        experiment_name="seed-42",
        workspace="lab",
        config={"seed": 42},
    )

    assert callback.kwargs == {
        "project": "medical-vqa",
        "experiment_name": "seed-42",
        "workspace": "lab",
        "config": {"seed": 42},
    }
