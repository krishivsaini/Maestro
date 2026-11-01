"""Usage limits for the public demo.

The deployed demo runs on one free-tier Gemini key shared by every visitor; a single
script hammering ``POST /run`` would spend everyone's daily quota in minutes. Two
limits protect it:

- **Per visitor:** at most ``run_limit_per_ip`` runs per ``run_limit_window_minutes``
  from one IP. Runs on a visitor's own key spend their quota, not the shared one, so
  they get ``byok_limit_multiplier`` times as many (still capped, against abuse).
- **Shared daily cap:** at most ``shared_daily_runs`` runs per UTC day on the server's
  key, across all visitors. Runs on a visitor's own key don't count.

Both are in memory, which is right for a single instance (the free Render service);
several instances would need a shared store such as Redis.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional


@dataclass
class Refusal:
    reason: str        # "visitor" | "daily"
    message: str
    retry_after: int   # seconds


class SlidingWindow:
    """At most ``limit`` events per ``window`` seconds per key."""

    def __init__(self, limit: int, window: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit, self.window, self.clock = limit, window, clock
        self._events: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def take(self, key: str) -> Optional[int]:
        """Record an event, or return the seconds until one is allowed."""
        now = self.clock()
        with self._lock:
            q = self._events[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return max(1, int(self.window - (now - q[0])) + 1)
            q.append(now)
            return None


class DailyCap:
    """At most ``limit`` events per UTC day."""

    def __init__(self, limit: int, today: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.limit, self.today = limit, today
        self._day, self._count = None, 0
        self._lock = threading.Lock()

    def take(self) -> Optional[int]:
        now = self.today()
        with self._lock:
            if now.date() != self._day:
                self._day, self._count = now.date(), 0
            if self._count >= self.limit:
                midnight = datetime(now.year, now.month, now.day, tzinfo=now.tzinfo).timestamp() + 86400
                return max(1, int(midnight - now.timestamp()))
            self._count += 1
            return None


def _minutes(seconds: int) -> str:
    m = max(1, round(seconds / 60))
    return f"{m} minute{'s' if m != 1 else ''}" if m < 90 else f"{round(m / 60)} hours"


class RunLimits:
    def __init__(self, settings, *, clock=time.monotonic, today=None) -> None:
        window = settings.run_limit_window_minutes * 60
        self.per_ip = settings.run_limit_per_ip
        self.shared = SlidingWindow(settings.run_limit_per_ip, window, clock) if settings.run_limit_per_ip else None
        byok = settings.run_limit_per_ip * settings.byok_limit_multiplier
        self.byok = SlidingWindow(byok, window, clock) if byok else None
        self.daily = DailyCap(settings.shared_daily_runs, **({"today": today} if today else {})) \
            if settings.shared_daily_runs else None
        self.window_minutes = settings.run_limit_window_minutes

    def check(self, ip: str, *, own_key: bool) -> Optional[Refusal]:
        """Count one run for ``ip``, or say why it is refused."""
        limiter = self.byok if own_key else self.shared
        wait = limiter.take(ip) if limiter else None
        if wait:
            n = limiter.limit
            return Refusal("visitor", f"You've reached this demo's limit of {n} run{'s' if n != 1 else ''} per "
                           f"{_minutes(self.window_minutes * 60)}. Try again in {_minutes(wait)}"
                           + ("." if own_key else ", or add your own free Gemini API key in Options."), wait)
        if not own_key and self.daily:
            wait = self.daily.take()
            if wait:
                return Refusal("daily", "The demo's shared key has used today's runs. Add your own free Gemini "
                               f"API key in Options to keep going, or try again in {_minutes(wait)}.", wait)
        return None
