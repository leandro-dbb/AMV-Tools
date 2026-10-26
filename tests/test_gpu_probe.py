"""First-run GPU suggestion (backend/gpu_probe.py): pure decision table plus
a probe that must never raise on a machine without any GPU tooling."""
from backend import gpu_probe
from backend.gpu_probe import recommend


def _nv(name, driver, cap):
    return [{"name": name, "driver": driver, "compute_cap": cap}]


def test_recommend_matrix():
    assert recommend([], [], "Windows", "AMD64")[0] == "cpu"
    assert recommend([], ["Microsoft Basic Display Adapter"], "Windows", "AMD64")[0] == "cpu"
    assert recommend([], ["AMD Radeon RX 6700 XT"], "Windows", "AMD64")[0] == "dml"
    assert recommend([], ["Intel(R) Arc(TM) A770 Graphics"], "Windows", "AMD64")[0] == "dml"
    assert recommend(_nv("NVIDIA GeForce RTX 4070", "581.57", 8.9), [], "Windows", "AMD64")[0] == "cu130"
    assert recommend(_nv("NVIDIA GeForce RTX 3060", "560.94", 8.6), [], "Windows", "AMD64")[0] == "cu126"
    assert recommend(_nv("NVIDIA GeForce GTX 1060", "581.57", 6.1), [], "Windows", "AMD64")[0] == "cu126"
    # Old driver that cannot report compute_cap: driver version decides.
    assert recommend(_nv("NVIDIA GeForce GTX 970", "472.12", None), [], "Windows", "AMD64")[0] == "cu126"
    assert recommend([], [], "Darwin", "arm64")[0] == "mps"
    assert recommend([], [], "Darwin", "x86_64")[0] == "cpu"
    assert recommend([], [], "Linux", "x86_64")[0] == "cpu"


def test_probe_without_tools(monkeypatch):
    monkeypatch.setattr(gpu_probe.shutil, "which", lambda _name: None)
    monkeypatch.setattr(gpu_probe, "_run", lambda _cmd: None)
    out = gpu_probe.probe()
    assert out["recommended"] in ("cpu", "mps")
    assert out["nvidia"] == []


def test_nvidia_smi_parsing(monkeypatch):
    monkeypatch.setattr(gpu_probe.shutil, "which", lambda _name: "nvidia-smi")
    monkeypatch.setattr(gpu_probe, "_run",
                        lambda _cmd: "NVIDIA GeForce RTX 3090, 580.95.05, 24576, 8.6\n")
    gpus = gpu_probe._nvidia_gpus()
    assert gpus == [{"vendor": "nvidia", "name": "NVIDIA GeForce RTX 3090", "driver": "580.95.05",
                     "vram_gb": 24.0, "compute_cap": 8.6}]
