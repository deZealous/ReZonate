"""A small polite HTTP client: identified User-Agent, per-host rate limit, disk cache, retries.

All third-party APIs in this project go through this, so etiquette rules
(MusicBrainz: <=1 req/s and a contactable User-Agent; LRCLIB: identify the
client and honour Retry-After on 429/503) are enforced in one place.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse

import requests


class PoliteClient:
    def __init__(self, user_agent: str, cache_dir: Path | None = None,
                 min_interval: dict[str, float] | None = None, max_retries: int = 4,
                 timeout: float = 30.0):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = user_agent
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval or {}
        self.max_retries = max_retries
        self.timeout = timeout
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def _wait(self, host: str) -> None:
        gap = self.min_interval.get(host, 0.0)
        with self._lock:
            now = time.monotonic()
            delay = self._last.get(host, 0.0) + gap - now
            if delay > 0:
                time.sleep(delay)
            self._last[host] = time.monotonic()

    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")

    def get_json(self, url: str, params: dict | None = None, *, cache: bool = True,
                 secret_params: tuple[str, ...] = ("api_key", "client_id")) -> dict | list | None:
        """GET JSON. Returns None on 404. Secret params are excluded from the cache key."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        public = {k: v for k, v in params.items() if k not in secret_params}
        key_url = f"{url}?{urlencode(sorted(public.items()))}"
        cp = self._cache_path(key_url) if cache else None
        if cp and cp.exists():
            return json.loads(cp.read_text())
        host = urlparse(url).netloc
        for attempt in range(self.max_retries + 1):
            self._wait(host)
            r = self.s.get(url, params=params, timeout=self.timeout)
            if r.status_code == 404:
                out = None
                break
            if r.status_code in (429, 503) or r.status_code >= 500:
                retry_after = r.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit()
                           else 2 ** attempt)
                continue
            r.raise_for_status()
            out = r.json()
            break
        else:
            raise RuntimeError(f"Giving up on {url} after {self.max_retries} retries")
        if cp:
            cp.write_text(json.dumps(out))
        return out


class HTTPRangeFile:
    """Read-only, seekable file object over HTTP Range requests with a block cache.

    Lets ``zipfile`` read the central directory and selected members of a remote
    multi-GB archive (e.g. fma_large.zip) without downloading the whole thing.
    """

    def __init__(self, url: str, session: requests.Session | None = None,
                 block_size: int = 1 << 20, max_blocks: int = 64):
        self.url = url
        self.s = session or requests.Session()
        head = self.s.head(url, allow_redirects=True, timeout=30)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        self.pos = 0
        self.block_size = block_size
        self.max_blocks = max_blocks
        self._blocks: dict[int, bytes] = {}

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self._blocks:
            if len(self._blocks) >= self.max_blocks:
                self._blocks.pop(next(iter(self._blocks)))
            start = i * self.block_size
            end = min(start + self.block_size, self.size) - 1
            for attempt in range(5):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            self._blocks[i] = r.content
        return self._blocks[i]

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            b, off = divmod(self.pos, self.block_size)
            chunk = self._block(b)[off: off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def close(self) -> None:
        self._blocks.clear()
