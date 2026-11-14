r"""
Download ALL missing WildChat shards from ModelScope with 4-stream parallel
range requests, per shard, with retry + resume + verification.

Skips shards already present and size-verified.  Handles the SSL-failing
shard (00000) by retrying more aggressively and falling back to a plain
single-stream download.

Verification per shard: byte size, PAR1 magic head/tail, pyarrow row count.

Usage:  python download_all_shards.py [--shards 0,1,2] [--nstream 4]
"""
import os
import sys
import time
import json
import argparse
import threading

import requests

BASE_MS = "https://modelscope.cn/datasets/allenai/WildChat-1M/resolve/master/"
OUTDIR = "data"
PART = os.path.join(OUTDIR, ".parts")
NSHARDS = 14
LOG = "download.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def session():
    s = requests.Session()
    s.trust_env = False
    s.proxies = {"http": "", "https": ""}
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s


def remote_size(url, tries=8):
    for i in range(tries):
        try:
            r = session().head(url, timeout=(10, 20), allow_redirects=True)
            if r.status_code == 200 and r.headers.get("content-length"):
                return int(r.headers["content-length"])
        except Exception as e:
            log(f"    head retry {i+1}/{tries}: {type(e).__name__}: {str(e)[:70]}")
        time.sleep(min(2 + 2 * i, 12))
    return None


def fetch_range(url, start, end, dst, res, key, tries=8):
    want = end - start + 1
    for attempt in range(tries):
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        if have >= want:
            res[key] = have
            return
        try:
            r = session().get(url, stream=True, timeout=(15, 40), allow_redirects=True,
                              headers={"Range": f"bytes={start+have}-{end}"})
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code}")
            with open(dst, "ab" if have else "wb") as f:
                for c in r.iter_content(1 << 16):
                    if c:
                        f.write(c)
            r.close()
        except Exception as e:
            log(f"      {key} attempt {attempt+1}/{tries} {type(e).__name__}: {str(e)[:60]}")
            time.sleep(min(2 + 2 * attempt, 15))
    res[key] = os.path.getsize(dst) if os.path.exists(dst) else 0


def verify(path, expected):
    """Return (ok, info)."""
    info = {"bytes": os.path.getsize(path)}
    if info["bytes"] != expected:
        info["err"] = f"size {info['bytes']} != {expected}"
        return False, info
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(-4, os.SEEK_END)
        tail = f.read(4)
    if head != b"PAR1" or tail != b"PAR1":
        info["err"] = f"magic head={head!r} tail={tail!r}"
        return False, info
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        info["rows"] = pf.metadata.num_rows
        info["rowgroups"] = pf.num_row_groups
    except Exception as e:
        info["err"] = f"pyarrow: {type(e).__name__}: {str(e)[:80]}"
        return False, info
    return True, info


def download_shard(idx, nstream):
    name = f"train-{idx:05d}-of-00014.parquet"
    rel = f"data/{name}"
    url = BASE_MS + rel
    dst = os.path.join(OUTDIR, name)

    # already present & valid?
    if os.path.exists(dst):
        ok, info = verify(dst, os.path.getsize(dst))
        if ok and info.get("rows", 0) > 0:
            log(f"  SKIP {name} already present ({info['bytes']:,} B, {info['rows']:,} rows)")
            return {"shard": idx, "status": "already", **info}

    size = remote_size(url)
    if size is None:
        log(f"  FAIL {name}: cannot get size")
        return {"shard": idx, "status": "fail_size"}

    log(f"  {name}: {size/1024/1024:.1f} MB, {nstream} streams")
    t0 = time.time()
    chunk = size // nstream
    ranges = []
    for i in range(nstream):
        st = i * chunk
        en = size - 1 if i == nstream - 1 else st + chunk - 1
        ranges.append((st, en, os.path.join(PART, f"{name}.part{i}")))

    res = {}
    ths = []
    for i, (st, en, p) in enumerate(ranges):
        t = threading.Thread(target=fetch_range, args=(url, st, en, p, res, f"p{i}"))
        t.start()
        ths.append(t)
    for t in ths:
        t.join()
    dt = time.time() - t0

    got = sum(res.values())
    if got != size:
        log(f"  FAIL {name}: got {got} of {size}")
        return {"shard": idx, "status": "fail_partial", "got": got, "want": size}

    tmp = dst + ".merge"
    with open(tmp, "wb") as out:
        for _, _, p in ranges:
            with open(p, "rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
    os.replace(tmp, dst)
    for _, _, p in ranges:
        os.remove(p)

    ok, info = verify(dst, size)
    mb = size / 1024 / 1024
    log(f"  {'OK  ' if ok else 'BAD '} {name} {mb:.1f} MB in {dt:.0f}s "
        f"({mb/max(dt,1):.2f} MB/s) rows={info.get('rows')} {info.get('err','')}")
    return {"shard": idx, "status": "ok" if ok else "bad", "seconds": round(dt, 1),
            **info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=None, help="comma list, default all")
    ap.add_argument("--nstream", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(PART, exist_ok=True)

    idxs = ([int(x) for x in args.shards.split(",")] if args.shards
            else list(range(NSHARDS)))
    log(f"===== download start: {len(idxs)} shards, {args.nstream} streams =====")

    results = []
    t_all = time.time()
    for idx in idxs:
        try:
            results.append(download_shard(idx, args.nstream))
        except Exception as e:
            log(f"  EXC shard {idx}: {type(e).__name__}: {str(e)[:120]}")
            results.append({"shard": idx, "status": "exception"})

    total_s = time.time() - t_all
    ok = [r for r in results if r["status"] in ("ok", "already")]
    log(f"===== download done: {len(ok)}/{len(idxs)} good in {total_s/60:.1f} min =====")
    for r in results:
        log(f"   shard {r['shard']:2d}: {r['status']:14s} "
            f"{r.get('rows','?')} rows {r.get('err','')}")

    with open("download_result.json", "w") as f:
        json.dump(results, f, indent=2)

    # final inventory
    import glob
    files = sorted(glob.glob(os.path.join(OUTDIR, "*.parquet")))
    tot = sum(os.path.getsize(p) for p in files)
    log(f"inventory: {len(files)} files, {tot/1024**3:.2f} GB")
    sys.exit(0 if len(ok) == len(idxs) else 1)


if __name__ == "__main__":
    main()
