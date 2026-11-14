r"""
Download ONE WildChat shard from ModelScope using parallel range requests,
then verify size + parquet magic.

Usage:
  python download_one_shard.py --shard 8

Checks performed after download:
  * byte size matches Content-Length
  * first 4 bytes == b"PAR1" (parquet magic)
  * last 4 bytes == b"PAR1"
  * optionally reads footer metadata with pyarrow to prove it is a valid file
"""
import os
import sys
import time
import json
import argparse
import threading
import requests

BASE = "https://modelscope.cn/datasets/allenai/WildChat-1M/resolve/master/"
OUTDIR = "data"
NSTREAM = 4
PART = os.path.join(OUTDIR, ".parts")


def make_session():
    s = requests.Session()
    s.trust_env = False
    s.proxies = {"http": "", "https": ""}
    return s


def head_size(url, tries=4):
    for i in range(tries):
        try:
            s = make_session()
            r = s.head(url, timeout=(10, 20), allow_redirects=True,
                       headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                cl = r.headers.get("content-length")
                if cl:
                    return int(cl), r.url
            time.sleep(1.5 * (i + 1))
        except Exception as e:
            print(f"  head retry {i+1}: {type(e).__name__}: {str(e)[:70]}", flush=True)
            time.sleep(1.5 * (i + 1))
    return None, None


def fetch_range(url, start, end, dst, results, key):
    """Download [start, end] into dst; retry with resume on failure."""
    want = end - start + 1
    for attempt in range(6):
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        if have >= want:
            results[key] = {"bytes": have, "ok": True, "attempts": attempt}
            return
        s = make_session()
        try:
            r = s.get(url, stream=True, timeout=(10, 25), allow_redirects=True,
                      headers={"Range": f"bytes={start+have}-{end}",
                               "User-Agent": "Mozilla/5.0"})
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code}")
            mode = "ab" if have else "wb"
            with open(dst, mode) as f:
                for chunk in r.iter_content(1 << 16):
                    if chunk:
                        f.write(chunk)
            r.close()
        except Exception as e:
            print(f"    {key} attempt {attempt+1} failed: "
                  f"{type(e).__name__}: {str(e)[:70]}", flush=True)
            time.sleep(1.5 * (attempt + 1))
    results[key] = {"bytes": os.path.getsize(dst) if os.path.exists(dst) else 0,
                    "ok": False, "attempts": 6}


def verify(path, expected_size):
    info = {"path": path, "bytes": os.path.getsize(path)}
    info["size_ok"] = info["bytes"] == expected_size
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(-4, os.SEEK_END)
        tail = f.read(4)
    info["head_magic"] = head.decode("latin1")
    info["tail_magic"] = tail.decode("latin1")
    info["magic_ok"] = head == b"PAR1" and tail == b"PAR1"
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        info["parquet_rows"] = pf.metadata.num_rows
        info["parquet_rowgroups"] = pf.num_row_groups
        info["parquet_ok"] = True
    except Exception as e:
        info["parquet_ok"] = False
        info["parquet_err"] = f"{type(e).__name__}: {str(e)[:100]}"
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(PART, exist_ok=True)

    name = f"train-{args.shard:05d}-of-00014.parquet"
    rel = f"data/{name}"
    url = BASE + rel
    dst = os.path.join(OUTDIR, name)

    print(f"=== shard {args.shard}: {name} ===", flush=True)
    size, final_url = head_size(url)
    if not size:
        print("FATAL: could not determine size", flush=True)
        sys.exit(1)
    print(f"  content-length = {size:,} bytes ({size/1024/1024:.1f} MB)", flush=True)
    print(f"  final host     = {final_url.split('/')[2] if final_url else '?'}", flush=True)

    chunk = size // NSTREAM
    ranges = []
    for i in range(NSTREAM):
        st = i * chunk
        en = size - 1 if i == NSTREAM - 1 else st + chunk - 1
        ranges.append((st, en, os.path.join(PART, f"{name}.part{i}")))

    print(f"  downloading with {NSTREAM} streams...", flush=True)
    t0 = time.time()
    results = {}
    threads = []
    for i, (st, en, p) in enumerate(ranges):
        t = threading.Thread(target=fetch_range, args=(url, st, en, p, results, f"part{i}"))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    dt = time.time() - t0

    got = sum(r["bytes"] for r in results.values())
    mb = got / 1024 / 1024
    print(f"\n  elapsed {dt:.1f}s  got {mb:.1f} MB  => {mb/max(dt,0.01):.2f} MB/s",
          flush=True)
    for k in sorted(results):
        print(f"    {k}: {results[k]}", flush=True)

    if got != size:
        print(f"FATAL: size mismatch (got {got}, want {size})", flush=True)
        sys.exit(1)

    # concatenate
    print("\n  merging parts...", flush=True)
    with open(dst, "wb") as out:
        for _, _, p in ranges:
            with open(p, "rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    out.write(b)

    info = verify(dst, size)
    print("\n=== VERIFY ===", flush=True)
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)

    if info.get("size_ok") and info.get("magic_ok") and info.get("parquet_ok"):
        for _, _, p in ranges:
            os.remove(p)
        print(f"\nOK  {dst}", flush=True)
        print(f"    {info['parquet_rows']:,} rows, "
              f"{info['parquet_rowgroups']} row groups", flush=True)
    else:
        print("\nFAILED verification - parts kept for inspection", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
