"""Live progress bars for the CLI (user-friendly requirement).

tqdm is used when importable; otherwise a minimal in-place \r writer is
used so every long-running command (models fetch, bench, eval,
swap-video) still shows a live bar plus ETA. Realtime FPS printed via
`live_line`.
"""
from __future__ import annotations

import sys
import time
from typing import Iterable, Optional

try:
    from tqdm import tqdm as _tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_TQDM = False


class FallbackBar:
    """Minimal terminal progress bar: [██████░░░░] 60% 12.3s ETA 8.1s"""

    WIDTH = 38

    def __init__(self, total: Optional[int], desc: str = "") -> None:
        self.total = total or 0
        self.desc = desc
        self.n = 0
        self._t0 = time.time()
        self._stream = sys.stderr

    def update(self, n: int = 1) -> None:
        self.n += n
        self._render()

    def _render(self) -> None:
        elapsed = max(time.time() - self._t0, 1e-9)
        if self.total > 0:
            frac = min(self.n / self.total, 1.0)
            filled = int(self.WIDTH * frac)
            bar = "█" * filled + "░" * (self.WIDTH - filled)
            rate = self.n / elapsed
            eta = (self.total - self.n) / rate if rate > 0 else 0.0
            text = f"\r[{bar}] {frac * 100:5.1f}%  {self.n}/{self.total}  {self.desc}  ETA {eta:4.1f}s"
        else:
            text = f"\r{self.n}  {self.desc}  {elapsed:5.1f}s"
        self._stream.write(text.ljust(120))
        self._stream.flush()

    def close(self) -> None:
        self._render()
        self._stream.write("\n")
        self._stream.flush()

    def __enter__(self) -> "FallbackBar":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def progress(iterable: Optional[Iterable] = None, total: Optional[int] = None, desc: str = ""):
    """Return a tqdm bar or a FallbackBar, transparently."""
    if _HAS_TQDM:
        return _tqdm(iterable, total=total, desc=desc, unit="", bar_format="{l_bar}{bar}| {percentage:5.1f}% {n_fmt}/{total_fmt} {desc} ETA {remaining}")  # type: ignore
    if total is None and iterable is not None:
        items = list(iterable)
        bar = FallbackBar(total=len(items), desc=desc)
        return _FallbackIterable(items, bar)
    return FallbackBar(total=total, desc=desc)


class _FallbackIterable:
    def __init__(self, items, bar: FallbackBar) -> None:
        self.items = items
        self.bar = bar

    def __iter__(self):
        try:
            for it in self.items:
                yield it
                self.bar.update(1)
        finally:
            self.bar.close()


def live_line(text: str) -> None:
    """Overwrite the current terminal line (realtime FPS display)."""
    sys.stderr.write("\r" + text.ljust(120))
    sys.stderr.flush()


def end_live_line() -> None:
    sys.stderr.write("\n")
    sys.stderr.flush()


def has_tqdm() -> bool:
    return _HAS_TQDM
