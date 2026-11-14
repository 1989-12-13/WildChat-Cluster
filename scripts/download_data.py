"""
Download all WildChat-1M parquet shards into ./data (relative to project root).

HF throttles parallel connections aggressively, so shards are fetched
SEQUENTIALLY over a single keep-alive connection.  Features:
  - resume via HTTP Range (.part files)
  - size + parquet magic verification against the HF API manifest
  - automatic fallback to a China-friendly HF mirror on repeated failure
  - retries with backoff

Run:  python _work/download_data.py
"""
import os
import sys
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# This box has a local HTTP proxy (127.0.0.1:7891) enabled via env vars.
# The CN mirror is domestic; going through the proxy measured ~2x SLOWER
# (1.8 MB/s vs 3.8 MB/s) and broke concurrent connections.  Bypass it.
SESSION = requests.Session()
SESSION.trust_env = False
SESSION.proxies = {"http": "", "https": ""}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MANIFEST = os.path.join(ROOT, "_work", "shards.json")

# User requested: use a China-friendly mirror only (no direct HF).
# hf-mirror.com measured slightly faster; alpha-ollama is the fallback.
BASES = [
    "https://hf-mirror.com/datasets/allenai/WildChat-1M/resolve/main/",
    "https://alpha-ollama.hf-mirror.com/datasets/allenai/WildChat-1M/resolve/main/",
]

RETRIES = 6
CHUNK = 1 << 20  # 1 MB
MAX_WORKERS = 2  # modest parallelism: mirrors tolerate 2, HF direct did not 4

os.makedirs(DATA_DIR, exist_ok=True)
with open(MANIFEST, "r", encoding="utf-8") as fh:
    shards = json.load(fh)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_one(shard):
    rel = shard["path"]
    name = os.path.basename(rel)
    expected = shard["size"]
    dest = os.path.join(DATA_DIR, name)
    part = dest + ".part"

    if os.path.exists(dest) and os.path.getsize(dest) == expected:
        return "skip"

    t0 = time.time()
    last_err = None
    for attempt in range(RETRIES):
        base = BASES[attempt % len(BASES)]
        url = base + rel
        pos = os.path.getsize(part) if os.path.exists(part) else 0
        if pos > expected:
            os.remove(part)
            pos = 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if pos:
            headers["Range"] = f"bytes={pos}-"
        mode = "ab" if pos else "wb"
        got_start = pos
        try:
            r = SESSION.get(url, headers=headers, stream=True, timeout=(20, 180))
            if r.status_code == 200 and pos:
                pos = 0
                mode = "wb"
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code} from {url}")
            with open(part, mode) as f:
                for chunk in r.iter_content(CHUNK):
                    if chunk:
                        f.write(chunk)
                        pos += len(chunk)
            r.close()
            got = os.path.getsize(part)
            if got != expected:
                raise RuntimeError(f"size mismatch: got {got}, expected {expected}")
            with open(part, "rb") as f:
                if f.read(4) != b"PAR1":
                    raise RuntimeError("bad parquet magic")
            os.replace(part, dest)
            dt = time.time() - t0
            log(f"  OK   {name}  {expected/1024/1024:.1f} MB  {dt:.1f}s  "
                f"({(expected-got_start)/1024/1024/max(dt,0.1):.1f} MB/s)")
            return "ok"
        except Exception as e:
            last_err = e
            log(f"  retry {attempt+1}/{RETRIES} {name}: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(2 * (attempt + 1))
    log(f"  FAIL {name}: {type(last_err).__name__}: {last_err}")
    return "fail"


def main():
    log(f"downloading {len(shards)} shards from CN mirror "
        f"({MAX_WORKERS} workers, resume enabled) -> {DATA_DIR}")
    total = sum(s["size"] for s in shards)
    log(f"total {total/1024/1024/1024:.3f} GB")
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_one, s): os.path.basename(s["path"]) for s in shards}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    log(f"elapsed {time.time()-t0:.0f}s")

    ok = sum(1 for v in results.values() if v in ("ok", "skip"))
    log(f"==== {ok}/{len(shards)} shards complete ====")
    missing = []
    for s in shards:
        p = os.path.join(DATA_DIR, os.path.basename(s["path"]))
        if not os.path.exists(p) or os.path.getsize(p) != s["size"]:
            missing.append(os.path.basename(s["path"]))
    if missing:
        log("MISSING/INCOMPLETE: " + ", ".join(missing))
        sys.exit(1)
    log("ALL SHARDS VERIFIED")


if __name__ == "__main__":
    main()
