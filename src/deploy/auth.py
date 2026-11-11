"""Password login for the browser API: signed expiring tokens and a login rate limit.

Standard library only. A token is ``<payload>.<signature>``, both base64url: the payload is JSON
``{"exp": <unix seconds>}`` and the signature its HMAC-SHA256 under the server's signing key.
"""

import base64
import binascii
import hashlib
import hmac
import json
from collections import defaultdict, deque


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(key: bytes, payload: str) -> str:
    return _b64(hmac.new(key, payload.encode(), hashlib.sha256).digest())


def issue_token(key: bytes, ttl_s: int, now: float) -> tuple[str, int]:
    """Return a token valid until ``now + ttl_s`` and that expiry (unix seconds)."""
    exp = int(now) + ttl_s
    payload = _b64(json.dumps({"exp": exp}).encode())
    return f"{payload}.{_sign(key, payload)}", exp


def verify_token(key: bytes, token: str, now: float) -> bool:
    """True when ``token`` was signed with ``key`` and has not expired."""
    parts = token.split(".")
    if len(parts) != 2 or not all(parts):
        return False
    payload, sig = parts
    try:
        if not hmac.compare_digest(_sign(key, payload), sig):
            return False
        exp = json.loads(_unb64(payload))["exp"]
    except (ValueError, KeyError, TypeError, binascii.Error, UnicodeError):
        return False
    return isinstance(exp, int) and now < exp


def check_password(given: str, expected: str) -> bool:
    """Constant-time comparison; an empty expected password never matches (fail closed)."""
    if not expected:
        return False
    return hmac.compare_digest(given.encode(), expected.encode())


class LoginLimiter:
    """Failed-login counter over a sliding window, per client and in total.

    In memory: the app runs in a single container, and a reset on restart is acceptable.
    """

    def __init__(self, max_failures: int = 5, window_s: float = 600, max_global_failures: int = 30):
        self.max_failures = max_failures
        self.window_s = window_s
        self.max_global_failures = max_global_failures
        self._per_client: dict[str, deque[float]] = defaultdict(deque)
        self._global: deque[float] = deque()

    def _prune(self, times: deque[float], now: float) -> None:
        while times and times[0] <= now - self.window_s:
            times.popleft()

    def blocked(self, client: str, now: float) -> bool:
        self._prune(self._global, now)
        times = self._per_client.get(client)
        if times is not None:
            self._prune(times, now)
            if not times:  # keep the dict from growing with one-off clients
                del self._per_client[client]
                times = None
        n = len(times) if times else 0
        return n >= self.max_failures or len(self._global) >= self.max_global_failures

    def record_failure(self, client: str, now: float) -> None:
        self._per_client[client].append(now)
        self._global.append(now)
