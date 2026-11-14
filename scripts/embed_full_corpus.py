r"""
FULL-CORPUS embedding driver: all 14 WildChat shards x Qwen3-Embedding-4B
across N GPUs.

Architecture
------------
We cannot hold 14 x 117k = ~1.6M texts + 1.6M x 2560 x 4B (=16 GB) vectors in
RAM at once, so the unit of work is a SHARD:

  for each shard (sequentially):
      plan      -> extract user turns, tokenize, bucket by length
      embed     -> N GPU workers, each owning a balanced slice of the batches
      finalise  -> merge blocks -> per-shard vectors.npy + meta.parquet

Per-shard artifacts are kept, and a top-level manifest records what is done,
so the whole run is resumable at shard granularity (and at batch granularity
inside a shard).

Finally `combine` concatenates per-shard npy/parquet into full-corpus arrays.

Why per-shard rather than one giant pass:
  * bounded peak RAM (one shard's texts at a time)
  * natural resume point
  * if a shard fails, the other 13 survive
  * each shard is ~120k rows -> ~1.15 GB of vectors, easy to sanity check

Usage
-----
  python embed_full_corpus.py --stage run --gpus 0,1,2
  python embed_full_corpus.py --stage combine
  python embed_full_corpus.py --stage status
"""
import os
import re
import sys
import json
import time
import glob
import argparse
import subprocess

import numpy as np

ROOT = os.environ.get("WILDCHAT_ROOT", ".")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out_full")
LOGDIR = os.path.join(ROOT, "logs")
MODEL_DIR = "/home/public/models/Qwen3-Embedding-4B"

MAX_LEN = 8192
DIM = 2560
TARGET_TOKENS = 4096
MAX_BATCH = 64
BLOCK_ROWS = 4000

_SHARD_RE = re.compile(r"train-(\d{5})-of-00014\.parquet$")


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)


def shard_paths():
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, "*.parquet"))):
        m = _SHARD_RE.search(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return out


def shard_dir(idx):
    return os.path.join(OUT, f"shard_{idx:02d}")


# ======================================================================= plan
def plan_shard(idx, src):
    """Extract user turns of one shard, tokenize, bucket, persist plan."""
    from tokenizers import Tokenizer
    import pyarrow.parquet as pq

    sd = shard_dir(idx)
    work = os.path.join(sd, "work")
    os.makedirs(work, exist_ok=True)

    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tok.no_truncation()

    rows = []
    n_dropped = 0
    pf = pq.ParquetFile(src)
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["conversation", "language", "model", "timestamp"])
        convs = tbl.column("conversation").to_pylist()
        langs = tbl.column("language").to_pylist()
        models = tbl.column("model").to_pylist()
        tss = tbl.column("timestamp").to_pylist()
        for conv, clang, cmodel, cts in zip(convs, langs, models, tss):
            for ti, turn in enumerate(conv or []):
                if turn.get("role") != "user":
                    continue
                content = turn.get("content")
                if content is None or not str(content).strip():
                    n_dropped += 1
                    continue
                rows.append({
                    "turn_identifier": turn.get("turn_identifier"),
                    "content": content,
                    "lang": turn.get("language"),
                    "conv_lang": clang,
                    "model": cmodel,
                    "ts": cts.isoformat() if cts is not None else None,
                    "country": turn.get("country"),
                    "state": turn.get("state"),
                    "conv_turn_idx": ti,
                    "shard": idx,
                })

    # tokenize with the raw tokenizers API (faster than HF for 1-by-1)
    lens = np.empty(len(rows), dtype=np.int32)
    for i, r in enumerate(rows):
        ids = tok.encode(r["content"], add_special_tokens=True)
        lens[i] = min(len(ids), MAX_LEN)
    del tok

    order = np.argsort(lens, kind="stable")
    batches, cur, cur_max = [], [], 0
    for j in order:
        L = int(lens[j])
        new_max = max(cur_max, L)
        if cur and (len(cur) >= MAX_BATCH or new_max * (len(cur) + 1) > TARGET_TOKENS):
            batches.append(np.array(cur, dtype=np.int32))
            cur, cur_max, new_max = [], 0, L
        cur.append(int(j))
        cur_max = new_max
    if cur:
        batches.append(np.array(cur, dtype=np.int32))

    load = np.array([int(lens[b].sum()) for b in batches], dtype=np.int64)
    np.save(os.path.join(work, "lens.npy"), lens)
    np.savez_compressed(os.path.join(work, "batches.npz"),
                        **{f"b{i}": b for i, b in enumerate(batches)})
    with open(os.path.join(work, "rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "shard": idx, "src": os.path.basename(src),
        "n_items": len(rows), "n_dropped_empty": n_dropped,
        "n_batches": len(batches),
        "lens_mean": float(lens.mean()), "lens_median": float(np.median(lens)),
        "lens_p99": float(np.percentile(lens, 99)), "lens_max": int(lens.max()),
        "truncated_at_maxlen": int((lens >= MAX_LEN).sum()),
        "batch_load_total": int(load.sum()),
    }
    with open(os.path.join(work, "plan.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"  shard {idx}: {len(rows):,} turns, {len(batches):,} batches, "
        f"dropped {n_dropped:,} empty")
    return meta


# ====================================================================== worker
def worker_main(shard_idx, wid, device, gpu_budget=None, limit_batches=None):
    """Embed this worker's share of one shard's batches on one GPU."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    sd = shard_dir(shard_idx)
    work = os.path.join(sd, "work")
    blocks = os.path.join(sd, "blocks")
    os.makedirs(blocks, exist_ok=True)

    with open(os.path.join(work, "plan.json")) as f:
        plan = json.load(f)
    n_batches = plan["n_batches"]
    if limit_batches:                      # smoke-testing only
        n_batches = min(n_batches, limit_batches)

    lens = np.load(os.path.join(work, "lens.npy"))
    data = np.load(os.path.join(work, "batches.npz"))
    rows = []
    with open(os.path.join(work, "rows.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # balanced split by token load
    ids = np.arange(n_batches)
    load = np.array([int(lens[data[f"b{i}"]].sum()) for i in ids], dtype=np.int64)
    if gpu_budget is None:
        gpu_budget = 1
    # contiguous blocks of batches assigned round-robin by cumulative load
    order = np.argsort(-load)
    assign = np.zeros(n_batches, dtype=np.int32)
    acc = np.zeros(gpu_budget, dtype=np.int64)
    for i in order:
        k = int(np.argmin(acc))
        assign[i] = k
        acc[k] += load[i]
    my_batches = [int(i) for i in ids[assign == wid]]

    dev = device
    torch.cuda.set_device(dev)
    tag = f"s{shard_idx}w{wid}"
    log(f"[{tag}] device={dev} batches={len(my_batches)}/{n_batches} "
        f"tokens={load[assign==wid].sum():,}")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                          trust_remote_code=True)
    except TypeError:
        model = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16,
                                          trust_remote_code=True)
    model = model.to(dev).eval()
    log(f"[{tag}] model ready, VRAM {torch.cuda.memory_allocated(dev)/1024**3:.2f} GB")

    # resume
    state_path = os.path.join(blocks, f"state_w{wid}.json")
    done = set()
    blk = 0
    if os.path.exists(state_path):
        with open(state_path) as f:
            st = json.load(f)
        done = set(st.get("done", []))
        blk = st.get("next_blk", 0)
        log(f"[{tag}] resume: {len(done)} batches done")
    todo = [b for b in my_batches if b not in done]
    if not todo:
        log(f"[{tag}] nothing to do")
        return 0

    buf_v, buf_i, buf_b = [], [], []
    n_done = 0
    t0 = time.time()

    def save_state():
        with open(state_path, "w") as f:
            json.dump({"done": sorted(done), "next_blk": blk}, f)

    def flush():
        nonlocal blk, buf_v, buf_i, buf_b
        if not buf_v:
            return
        p = os.path.join(blocks, f"block_w{wid}_{blk:05d}.npz")
        np.savez(p, vec=np.concatenate(buf_v, axis=0),
                 idx=np.concatenate(buf_i).astype(np.int64))
        done.update(buf_b)
        blk += 1
        buf_v, buf_i, buf_b = [], [], []
        save_state()
        log(f"[{tag}] flushed {os.path.basename(p)} "
            f"batches {len(done)}/{len(my_batches)}")

    for bi in todo:
        b = data[f"b{bi}"]
        texts = [rows[int(i)]["content"] for i in b]
        enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc)
            h = out.last_hidden_state
            am = enc["attention_mask"]
            if bool((am[:, -1].sum() == am.shape[0]).item()):
                emb = h[:, -1]
            else:
                sl = am.sum(dim=1) - 1
                emb = h[torch.arange(h.shape[0], device=dev), sl]
            emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
        buf_v.append(emb.cpu().numpy())
        buf_i.append(np.asarray(b, dtype=np.int64))
        buf_b.append(bi)
        n_done += len(b)
        if sum(len(x) for x in buf_i) >= BLOCK_ROWS:
            flush()
        if len(buf_b) % 100 == 0:
            el = time.time() - t0
            log(f"[{tag}] {n_done:,} rows | {n_done/max(el,1):.1f} rows/s | "
                f"{len(done)}/{len(my_batches)} batches | {el:.0f}s")
    flush()
    log(f"[{tag}] DONE {n_done:,} rows in {time.time()-t0:.0f}s")
    return n_done


# ==================================================================== finalise
def finalise_shard(idx):
    import pyarrow as pa
    import pyarrow.parquet as pq

    sd = shard_dir(idx)
    work = os.path.join(sd, "work")
    blocks = os.path.join(sd, "blocks")
    with open(os.path.join(work, "plan.json")) as f:
        plan = json.load(f)
    n = plan["n_items"]

    files = sorted(glob.glob(os.path.join(blocks, "*.npz")))
    if not files:
        raise SystemExit(f"shard {idx}: no blocks")
    vs, ii = [], []
    for p in files:
        d = np.load(p)
        vs.append(d["vec"])
        ii.append(np.ravel(d["idx"]))
    vec = np.concatenate(vs, axis=0)
    idxs = np.concatenate(ii).astype(np.int64)
    if vec.shape[0] != n:
        raise SystemExit(f"shard {idx}: {vec.shape[0]} vectors vs {n} items")
    if len(np.unique(idxs)) != n:
        raise SystemExit(f"shard {idx}: duplicate indices")
    o = np.argsort(idxs, kind="stable")
    vec = vec[o]

    np.save(os.path.join(sd, "vectors.npy"), vec.astype(np.float32))

    lens = np.load(os.path.join(work, "lens.npy"))
    rows = []
    with open(os.path.join(work, "rows.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    tbl = pa.table({
        # 注意：此处 item_id 是「分片内局部行号」0..n-1。
        # 全局唯一的行号由 combine() 统一改写，不要在分片阶段当成全局 ID 使用。
        "item_id": np.arange(n, dtype=np.int64),
        "turn_identifier": pa.array([r["turn_identifier"] for r in rows], pa.int64()),
        "conv_turn_idx": pa.array([r["conv_turn_idx"] for r in rows], pa.int32()),
        "content": pa.array([r["content"] for r in rows], pa.string()),
        "lang": pa.array([r["lang"] for r in rows], pa.string()),
        "conv_lang": pa.array([r["conv_lang"] for r in rows], pa.string()),
        "model": pa.array([r["model"] for r in rows], pa.string()),
        "country": pa.array([r["country"] for r in rows], pa.string()),
        "state": pa.array([r["state"] for r in rows], pa.string()),
        "timestamp": pa.array([r["ts"] for r in rows], pa.string()),
        "token_len": pa.array(lens.astype(np.int32), pa.int32()),
        "truncated": pa.array(lens >= MAX_LEN),
    })
    pq.write_table(tbl, os.path.join(sd, "meta.parquet"), compression="zstd")
    log(f"  finalised shard {idx}: {vec.shape[0]:,} vectors, "
        f"{os.path.getsize(os.path.join(sd,'vectors.npy'))/1024**2:.0f} MB")
    return {"shard": idx, "n_vectors": int(vec.shape[0]),
            "truncated": int((lens >= MAX_LEN).sum())}


def run_shard(idx, src, gpus):
    """plan -> N workers -> finalise, for one shard."""
    sd = shard_dir(idx)
    os.makedirs(sd, exist_ok=True)
    planf = os.path.join(sd, "work", "plan.json")
    if not os.path.exists(planf):
        log(f"== shard {idx}: planning ==")
        plan_shard(idx, src)
    else:
        log(f"== shard {idx}: plan exists, reusing ==")

    log(f"== shard {idx}: embedding on {len(gpus)} GPU(s) {gpus} ==")
    procs, logfs = [], []
    for wid, dev in enumerate(gpus):
        lf = open(os.path.join(LOGDIR, f"embed_s{idx}_w{wid}.log"), "w")
        logfs.append(lf)
        p = subprocess.Popen(
            [sys.executable, __file__, "--stage", "_worker",
             "--shard", str(idx), "--wid", str(wid),
             "--device", dev, "--ngpu", str(len(gpus)),
             "--outdir", OUT],
            stdout=lf, stderr=subprocess.STDOUT)
        procs.append(p)
    codes = [p.wait() for p in procs]
    for lf in logfs:
        lf.close()
    if any(c != 0 for c in codes):
        raise SystemExit(f"shard {idx}: worker exit codes {codes}")
    return finalise_shard(idx)


# ====================================================================== combine
def combine():
    import pyarrow as pa
    import pyarrow.parquet as pq

    done = []
    for idx, _ in shard_paths():
        sd = shard_dir(idx)
        if os.path.exists(os.path.join(sd, "vectors.npy")) and \
           os.path.exists(os.path.join(sd, "meta.parquet")):
            done.append(idx)
    if not done:
        raise SystemExit("no finished shards")
    log(f"combining {len(done)} shards: {done}")

    total = 0
    for idx in done:
        t = pq.read_table(os.path.join(shard_dir(idx), "meta.parquet"), columns=["item_id"])
        total += t.num_rows
    log(f"total rows {total:,}")

    vec_path = os.path.join(OUT, "vectors.npy")
    fp = np.lib.format.open_memmap(vec_path, mode="w+", dtype=np.float32,
                                   shape=(total, DIM))
    off = 0
    metas = []
    for idx in done:
        v = np.load(os.path.join(shard_dir(idx), "vectors.npy"), mmap_mode="r")
        fp[off:off + v.shape[0]] = v
        m = pq.read_table(os.path.join(shard_dir(idx), "meta.parquet"))
        metas.append(m)
        off += v.shape[0]
        log(f"  appended shard {idx}: {v.shape[0]:,} rows (offset {off:,})")
    fp.flush()
    del fp

    full = pa.concat_tables(metas)

    # 修正 item_id：finalise_shard 为每个分片各自写 0..n-1（分片内局部行号），
    # 直接拼接会产生大量重复。这里统一改写为全局行号 0..total-1。
    idx_col = full.schema.get_field_index("item_id")
    full = full.set_column(idx_col, "item_id",
                           pa.array(np.arange(total, dtype=np.int64)))
    _ids = full.column("item_id").to_numpy()
    if len(np.unique(_ids)) != total or _ids.min() != 0 or _ids.max() != total - 1:
        raise SystemExit("item_id 修正失败：仍存在重复或不连续")
    log(f"item_id 已改写为全局行号 0..{total-1}")

    # turn_identifier 来自原始数据，应全局唯一；不唯一则说明分片有重叠
    if "turn_identifier" in full.column_names:
        _ti = full.column("turn_identifier").to_numpy()
        n_uniq = len(np.unique(_ti))
        if n_uniq != total:
            log(f"警告：turn_identifier 唯一值 {n_uniq:,} != 行数 {total:,}")
        else:
            log("turn_identifier 全局唯一，可用作跨表关联键")

    pq.write_table(full, os.path.join(OUT, "meta.parquet"), compression="zstd")

    manifest = {
        "model": "Qwen3-Embedding-4B", "pooling": "last_token",
        "max_length": MAX_LEN, "dimension": DIM, "dtype": "float32",
        "normalized": True, "instruct": None,
        "unit": "all user turns", "shards": done, "n_vectors": int(total),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log(f"combined -> {OUT}  ({total:,} x {DIM})")


# ======================================================================= status
def status():
    print(f"{'shard':>6} {'planned':>8} {'blocks':>7} {'final':>6} {'vectors':>9} {'trunc':>6}")
    for idx, src in shard_paths():
        sd = shard_dir(idx)
        planned = os.path.exists(os.path.join(sd, "work", "plan.json"))
        nb = len(glob.glob(os.path.join(sd, "blocks", "*.npz")))
        fin = os.path.exists(os.path.join(sd, "vectors.npy"))
        nv = tr = ""
        if fin:
            try:
                import pyarrow.parquet as pq
                t = pq.read_table(os.path.join(sd, "meta.parquet"),
                                  columns=["truncated"])
                nv = t.num_rows
                tr = sum(t.column("truncated").to_pylist())
            except Exception:
                pass
        print(f"{idx:>6} {str(planned):>8} {nb:>7} {str(fin):>6} {str(nv):>9} {str(tr):>6}")
    if os.path.exists(os.path.join(OUT, "vectors.npy")):
        v = np.load(os.path.join(OUT, "vectors.npy"), mmap_mode="r")
        print(f"\nCOMBINED: {v.shape}  "
              f"{os.path.getsize(os.path.join(OUT,'vectors.npy'))/1024**3:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["run", "combine", "status", "_worker"])
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--shards", default=None, help="comma list; default all")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="smoke test: only use the first N batches")
    ap.add_argument("--outdir", default=None, help="override OUT directory")
    args = ap.parse_args()

    if args.outdir:
        global OUT, LOGDIR
        OUT = args.outdir
        LOGDIR = os.path.join(OUT, "logs")

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)

    if args.stage == "status":
        status()
        return

    if args.stage == "_worker":
        worker_main(args.shard, args.wid, args.device, gpu_budget=args.ngpu,
                    limit_batches=args.limit_batches)
        return

    if args.stage == "combine":
        combine()
        return

    gpus = [f"cuda:{g}" for g in args.gpus.split(",")]
    allsh = shard_paths()
    if args.shards:
        want = {int(x) for x in args.shards.split(",")}
        allsh = [(i, p) for i, p in allsh if i in want]
    log(f"=== FULL RUN: {len(allsh)} shards x {len(gpus)} GPU(s) ===")
    t0 = time.time()
    results = []
    for idx, src in allsh:
        try:
            r = run_shard(idx, src, gpus)
            results.append(r)
            log(f"shard {idx} complete: {r}")
        except SystemExit as e:
            log(f"shard {idx} FAILED: {e}")
        except Exception as e:
            log(f"shard {idx} ERROR: {type(e).__name__}: {e}")
    log(f"=== all shards processed in {(time.time()-t0)/3600:.2f} h ===")
    with open(os.path.join(OUT, "run_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    log("now run: --stage combine")


if __name__ == "__main__":
    main()
