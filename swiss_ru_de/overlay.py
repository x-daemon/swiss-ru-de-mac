"""Always-on-top subtitle window (Tkinter, no extra install).

Drag it with the mouse, mouse wheel changes the font size, Esc or double-click closes.
"""

from __future__ import annotations

import queue
from typing import Callable

from .output import line_texts
from .pipeline import Line


class Overlay:
    def __init__(self, target: str, font_size: int = 22, opacity: float = 0.85,
                 status: str = "Listening …"):
        import tkinter as tk

        self.tk = tk
        self.target = target
        self.q: queue.Queue = queue.Queue()
        self.root = root = tk.Tk()
        root.title("swiss-ru-de")
        root.overrideredirect(True)  # no title bar
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", opacity)
        except tk.TclError:
            pass
        bg = "#101010"
        root.configure(bg=bg)

        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.width = int(sw * 0.7)
        self._x, self._bottom = (sw - self.width) // 2, sh - 80  # bottom edge stays put

        self.font_size = font_size
        self.small = tk.Label(root, text="", fg="#9a9a9a", bg=bg, wraplength=self.width - 40,
                              font=("Segoe UI", max(10, font_size * 2 // 3)), justify="center")
        self.main = tk.Label(root, text=status, fg="#ffffff", bg=bg, wraplength=self.width - 40,
                             font=("Segoe UI", font_size, "bold"), justify="center")
        self.small.pack(fill="x", padx=20, pady=(8, 0))
        self.main.pack(fill="both", expand=True, padx=20, pady=(0, 8))

        for w in (root, self.small, self.main):
            w.bind("<ButtonPress-1>", self._start_drag)
            w.bind("<B1-Motion>", self._drag)
            w.bind("<Double-Button-1>", lambda e: self.close())
            w.bind("<MouseWheel>", self._zoom)
            w.bind("<Button-4>", lambda e: self._set_font(self.font_size + 2))
            w.bind("<Button-5>", lambda e: self._set_font(self.font_size - 2))
        root.bind("<Escape>", lambda e: self.close())
        self._closed = False
        self._fit()

    # thread-safe API -------------------------------------------------------------
    def push(self, line: Line) -> None:
        self.q.put(("line", line))

    def status(self, text: str) -> None:
        self.q.put(("status", text))

    # ----------------------------------------------------------------------------
    def run(self, should_stop: Callable[[], bool] = lambda: False) -> None:
        def poll():
            if self._closed:
                return
            if should_stop():
                self.close()
                return
            try:
                while True:
                    kind, item = self.q.get_nowait()
                    if kind == "status":
                        self.small.config(text=item)
                    else:
                        texts = line_texts(item, self.target)
                        self.small.config(text=texts[-2] if len(texts) >= 2 else "")
                        self.main.config(text=texts[-1])
                    self._fit()
            except queue.Empty:
                pass
            self.root.after(100, poll)

        self.root.after(100, poll)
        self.root.mainloop()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.root.destroy()

    def _fit(self) -> None:
        """Grow/shrink the window to the text, keeping its bottom edge in place."""
        self.root.update_idletasks()
        h = self.root.winfo_reqheight()
        self.root.geometry(f"{self.width}x{h}+{self._x}+{self._bottom - h}")

    def _start_drag(self, e) -> None:
        self._dx, self._dy = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y()

    def _drag(self, e) -> None:
        self._x = e.x_root - self._dx
        self._bottom = e.y_root - self._dy + self.root.winfo_height()
        self.root.geometry(f"+{self._x}+{e.y_root - self._dy}")

    def _zoom(self, e) -> None:
        self._set_font(self.font_size + (2 if e.delta > 0 else -2))

    def _set_font(self, size: int) -> None:
        self.font_size = max(10, min(72, size))
        self.main.config(font=("Segoe UI", self.font_size, "bold"))
        self.small.config(font=("Segoe UI", max(10, self.font_size * 2 // 3)))
        self._fit()
