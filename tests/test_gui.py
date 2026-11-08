"""Window tests; skipped when Tkinter or a display is not available."""

import os
import sys
import threading
from dataclasses import replace

import pytest

tk = pytest.importorskip("tkinter")
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("no display", allow_module_level=True)

from swiss_ru_de import gui  # noqa: E402
from swiss_ru_de.settings import Settings, load_settings  # noqa: E402


class FakePipeline:
    stop_event = threading.Event()
    error = None
    dialect_transcriber = None

    def stop(self, wait=True):
        self.stop_event.set()


def fake_startup(self, file, url):
    self.pipeline = FakePipeline()
    line = self.book.new(1, 3, "Wir fangen an.", "ru", dialect=True)
    self.book.update(line.id, translation="Мы начинаем.", translation_pending=False)
    self.book.update(line.id, dialect="Mir fö aa.", dialect_pending=False)
    self.book.new(4, 6, "Noch Fragen?", "ru", dialect=True)  # stays pending


def last_row(pane):
    return pane.text.get("1.0", "end").strip().splitlines()[-1]


def test_two_panes_settings_and_export(monkeypatch, tmp_path):
    monkeypatch.setattr(gui.App, "_startup", fake_startup)
    sf = tmp_path / "settings.json"
    app = gui.App(Settings(), settings_file=sf)
    result = {}

    def check():
        try:
            app._poll()
            result["rows"] = (len(app.top.rows), len(app.bottom.rows))
            result["first"] = (app.top.text.get("1.0", "1.end"), app.bottom.text.get("1.0", "1.end"))
            result["pending"] = last_row(app.bottom)
            app.apply_settings(replace(app.s, target="de", top="german"))
            result["switched"] = (app.top.title.cget("text"), app.bottom.title.cget("text"),
                                  app.bottom.text.get("1.0", "1.end"))
            from tkinter import filedialog

            monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **kw: str(tmp_path / "t.srt"))
            app.save_transcript()
        finally:
            app.close()

    app.root.after(300, check)
    app.run()
    assert result["rows"] == (2, 2)
    assert "Mir fö aa." in result["first"][0] and "Мы начинаем." in result["first"][1]
    assert result["pending"].endswith("…")
    assert result["switched"] == ("HOCHDEUTSCH", "HOCHDEUTSCH", "00:01   Wir fangen an.")
    assert load_settings(sf).target == "de"
    assert "Wir fangen an." in (tmp_path / "t.srt").read_text(encoding="utf-8")
