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
    assert recommend(_nv("NVIDIA GeForce RTX 3060", "580.88", 8.6), [], "Windows", "AMD64")[0] == "cu126"
    assert recommend(_nv("NVIDIA GeForce RTX 3060", "580.95.05", 8.6), [], "Linux", "x86_64")[0] == "cu130"
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


def test_probe_names_the_apple_chip(monkeypatch):
    # The setup screen showed "Detected on this PC: no usable GPU" on an M1 Pro:
    # macOS has no adapter list, the chip name is the GPU to show.
    monkeypatch.setattr(gpu_probe.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gpu_probe.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(gpu_probe, "_run", lambda cmd: "Apple M1 Pro\n" if cmd[0] == "sysctl" else None)
    out = gpu_probe.probe()
    assert out["recommended"] == "mps"
    assert out["adapters"] == [{"name": "Apple M1 Pro", "vendor": "apple"}]
