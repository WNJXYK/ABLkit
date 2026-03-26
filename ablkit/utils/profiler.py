"""
Resource profiler for ABL experiments.

Tracks wall time, GPU memory, and CPU memory with phase-level breakdown.

Usage:
    from ablkit.utils.profiler import Profiler

    prof = Profiler()
    prof.start()

    with prof.phase("data_loading"):
        ...
    with prof.phase("training"):
        ...
    with prof.phase("testing"):
        ...

    prof.stop()
    prof.report()
"""

import os
import time
import platform
from contextlib import contextmanager

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import resource as _resource
except ImportError:
    _resource = None


def _get_rss_mb() -> float:
    """Get current RSS in MB."""
    if _resource is None:
        return 0.0
    try:
        rusage = _resource.getrusage(_resource.RUSAGE_SELF)
        if platform.system() == "Darwin":
            return rusage.ru_maxrss / (1024 ** 2)
        return rusage.ru_maxrss / 1024
    except Exception:
        return 0.0


def _get_gpu_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if _HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def _get_gpu_peak_mb() -> float:
    """Get peak GPU memory allocated in MB since last reset."""
    if _HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m)}m{s:.0f}s"


class _PhaseRecord:
    __slots__ = ("name", "wall_time", "gpu_start", "gpu_peak", "cpu_rss_start", "cpu_rss_end")

    def __init__(self, name):
        self.name = name
        self.wall_time = 0.0
        self.gpu_start = 0.0
        self.gpu_peak = 0.0
        self.cpu_rss_start = 0.0
        self.cpu_rss_end = 0.0

    @property
    def gpu_delta(self):
        return max(self.gpu_peak - self.gpu_start, 0.0)


class Profiler:
    """
    Lightweight profiler with optional phase-level breakdown.

    Phases are defined via context manager:
        with prof.phase("training"):
            ...

    Each phase records:
      - Wall time
      - GPU peak memory delta (peak during phase minus start of phase)
      - CPU RSS at end of phase
    """

    def __init__(self):
        self._start_time = None
        self._wall_time = 0.0
        self._peak_cpu_mb = 0.0
        self._peak_gpu_mb = 0.0
        self._gpu_available = _HAS_TORCH and torch.cuda.is_available()
        self._phases: list = []
        self._current_phase: _PhaseRecord = None

    def start(self):
        self._start_time = time.time()
        if self._gpu_available:
            torch.cuda.reset_peak_memory_stats()

    def stop(self):
        self._wall_time = time.time() - self._start_time
        self._peak_cpu_mb = _get_rss_mb()
        if self._gpu_available:
            self._peak_gpu_mb = _get_gpu_peak_mb()

    @contextmanager
    def phase(self, name: str):
        """Record a named phase with timing and memory tracking."""
        rec = _PhaseRecord(name)
        rec.cpu_rss_start = _get_rss_mb()
        if self._gpu_available:
            torch.cuda.reset_peak_memory_stats()
            rec.gpu_start = _get_gpu_mb()

        rec_start = time.time()
        self._current_phase = rec
        try:
            yield rec
        finally:
            rec.wall_time = time.time() - rec_start
            rec.cpu_rss_end = _get_rss_mb()
            if self._gpu_available:
                rec.gpu_peak = _get_gpu_peak_mb()
            self._phases.append(rec)
            self._current_phase = None

    # ── Properties ──

    @property
    def wall_time(self) -> float:
        return self._wall_time

    @property
    def peak_cpu_mb(self) -> float:
        return self._peak_cpu_mb

    @property
    def peak_gpu_mb(self) -> float:
        return self._peak_gpu_mb

    @property
    def phases(self):
        return list(self._phases)

    # ── Reporting ──

    def summary(self) -> str:
        parts = [f"Wall time: {_fmt_time(self._wall_time)}"]
        parts.append(f"Peak CPU mem: {self._peak_cpu_mb:.1f} MB")
        if self._gpu_available:
            parts.append(f"Peak GPU mem: {self._peak_gpu_mb:.1f} MB")
        return " | ".join(parts)

    def phase_table(self) -> str:
        """Format phase breakdown as a table."""
        if not self._phases:
            return ""
        lines = []
        header = f"  {'Phase':<20s} {'Time':>10s} {'CPU RSS':>12s}"
        if self._gpu_available:
            header += f" {'GPU peak':>12s}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for p in self._phases:
            row = f"  {p.name:<20s} {_fmt_time(p.wall_time):>10s} {p.cpu_rss_end:>10.1f} MB"
            if self._gpu_available:
                row += f" {p.gpu_delta:>10.1f} MB"
            lines.append(row)
        # Total
        lines.append("  " + "-" * (len(header) - 2))
        total_time = sum(p.wall_time for p in self._phases)
        row = f"  {'TOTAL':<20s} {_fmt_time(total_time):>10s} {self._peak_cpu_mb:>10.1f} MB"
        if self._gpu_available:
            row += f" {self._peak_gpu_mb:>10.1f} MB"
        lines.append(row)
        return "\n".join(lines)

    def report(self):
        from ablkit.utils import print_log
        print_log(f"[Profiler] {self.summary()}", logger="current")
        table = self.phase_table()
        if table:
            print_log(f"[Profiler] Phase breakdown:\n{table}", logger="current")
