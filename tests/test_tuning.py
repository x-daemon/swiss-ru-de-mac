from swiss_ru_de import asr, cli, tuning
from swiss_ru_de.tuning import choose


def test_fast_hardware_gets_precision():
    t = choose("auto", 0.4, "cuda")
    assert (t.profile, t.beam, t.dialect) == ("precise", 5, True)


def test_medium_cpu_runs_dialect_only_when_idle_with_longer_segments():
    t = choose("auto", 1.8, "cpu")
    assert (t.profile, t.beam, t.dialect) == ("balanced", 1, True)
    assert t.min_segment_s == 3.6  # 2 x pass time: recognition keeps up with speech


def test_slow_cpu_drops_the_dialect_line_and_keeps_up():
    t = choose("auto", 4.0, "cpu")
    assert (t.profile, t.dialect, t.min_segment_s) == ("fast", False, 8.0)
    assert "paused" in t.describe(4.0)


def test_user_can_force_a_profile():
    assert choose("precise", 6.0, "cpu").dialect is True
    assert choose("fast", 0.2, "cuda").dialect is False


def test_report_gives_advice(monkeypatch):
    monkeypatch.setattr(tuning, "gpu_names", lambda: ["NVIDIA GeForce RTX 3060"])
    monkeypatch.setattr(tuning, "cuda_devices", lambda: 0)
    text = "\n".join(tuning.report(3.0, "cpu"))
    assert "CUDA libraries are missing" in text and "3.0 s per pass" in text
    monkeypatch.setattr(tuning, "gpu_names", lambda: ["Intel(R) Iris(R) Xe Graphics"])
    assert "OpenVINO" in "\n".join(tuning.report(3.0, "cpu"))
    assert "fastest" in "\n".join(tuning.report(0.3, "cuda"))


def test_benchmark_command(monkeypatch, capsys):
    class Timed:
        def __init__(self, model, **kw):
            self.pass_seconds, self.device = 2.0, "cpu"

    monkeypatch.setattr(asr, "Transcriber", Timed)
    monkeypatch.setattr(tuning, "gpu_names", lambda: [])
    assert cli.main(["--benchmark", "--cpu"]) == 0
    out = capsys.readouterr().out
    assert "2.0 s per pass" in out and "balanced mode" in out
