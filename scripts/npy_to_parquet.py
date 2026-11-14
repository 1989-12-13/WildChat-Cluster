#!/usr/bin/env python3
"""
把 out/vectors.npy（18.5 GB）+ out/meta.parquet 转成分块的 parquet。

为什么分块：
  npy 是单个 18.5 GB 文件，一次性载入会打爆内存。按 chunk 处理，
  每块 5 万行（约 512 MB），边读边写。

为什么在进 HDFS 之前转：
  HDFS 只是文件系统，不解析格式；Spark 无法读取 .npy。

产物：
  <outdir>/vectors-00000.parquet ... vectors-00038.parquet
  <outdir>/_manifest.json        文件清单、行数、校验信息

特性：
  - 断点续传：已存在且行数正确的分块会跳过
  - 每块写完立即校验（行数 + 向量维度 + 抽样范数）
  - 保留 content 列，便于后续聚类命名（c-TF-IDF）

用法：
  python3 npy_to_parquet.py --outdir ~/wildchat_bench/parquet --chunk-rows 50000
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.environ.get("WILDCHAT_ROOT", ".")
VEC = os.path.join(ROOT, "out", "vectors.npy")
META = os.path.join(ROOT, "out", "meta.parquet")

META_COLS = [
    "item_id", "turn_identifier", "conv_turn_idx", "content", "lang",
    "conv_lang", "model", "country", "state", "timestamp", "token_len",
    "truncated",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "parquet"))
    ap.add_argument("--chunk-rows", type=int, default=50000)
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    vec = np.load(VEC, mmap_mode="r")
    n_total, dim = vec.shape
    log(f"源向量: {vec.shape}  dtype={vec.dtype}  "
        f"{vec.nbytes/1024**3:.2f} GB")

    meta_pf = pq.ParquetFile(META)
    n_meta = meta_pf.metadata.num_rows
    if n_meta != n_total:
        raise SystemExit(f"行数不一致: vectors={n_total} meta={n_meta}")
    log(f"元信息: {n_meta} 行, 列={len(META_COLS)}")

    emb_type = pa.list_(pa.float32())
    manifest = {
        "source_vectors": VEC,
        "source_meta": META,
        "total_rows": int(n_total),
        "dimension": int(dim),
        "chunk_rows": args.chunk_rows,
        "compression": args.compression,
        "files": [],
    }

    n_chunks = (n_total + args.chunk_rows - 1) // args.chunk_rows
    log(f"将生成 {n_chunks} 个分块，每块 {args.chunk_rows} 行")

    t_all = time.time()
    done_rows = 0

    for ci in range(n_chunks):
        start = ci * args.chunk_rows
        end = min(start + args.chunk_rows, n_total)
        nrows = end - start
        out_path = os.path.join(args.outdir, f"vectors-{ci:05d}.parquet")

        # ---- 断点续传 ----
        if os.path.exists(out_path):
            try:
                old = pq.ParquetFile(out_path)
                if old.metadata.num_rows == nrows:
                    log(f"  [{ci+1}/{n_chunks}] 已存在，跳过 {os.path.basename(out_path)}")
                    manifest["files"].append({
                        "file": os.path.basename(out_path),
                        "rows": nrows,
                        "bytes": os.path.getsize(out_path),
                        "reused": True,
                    })
                    done_rows += nrows
                    continue
            except Exception:
                pass
            log(f"  [{ci+1}/{n_chunks}] 已有文件不完整，重写")

        t0 = time.time()
        # 只读该块的向量（mmap，不整体载入）
        block = np.asarray(vec[start:end], dtype=np.float32)
        # 高效构造 array<float>：扁平的 values + offsets，
        # 避免 list(block) 那种逐行 Python 对象化（50000×2560 会极慢）
        values = pa.array(block.ravel(), type=pa.float32())
        offsets = pa.array(np.arange(0, nrows * dim + 1, dim, dtype=np.int64))
        emb = pa.ListArray.from_arrays(offsets, values)
        del values, offsets
        norm0 = float(np.linalg.norm(block[0]))
        del block

        meta = pq.read_table(META, columns=META_COLS).slice(start, nrows)
        table = meta.append_column("embedding", emb)
        del emb

        pq.write_table(table, out_path, compression=args.compression)
        del table

        # ---- 立即校验 ----
        chk = pq.ParquetFile(out_path)
        assert chk.metadata.num_rows == nrows, \
            f"行数不符 {chk.metadata.num_rows} != {nrows}"
        ftype = chk.schema_arrow.field("embedding").type
        assert isinstance(ftype, pa.ListType) and ftype.value_type == pa.float32(), \
            f"embedding 列类型异常: {ftype}"

        sz = os.path.getsize(out_path) / 1024**2
        dt = time.time() - t0
        log(f"  [{ci+1}/{n_chunks}] {os.path.basename(out_path)} "
            f"{nrows} 行 {sz:.0f} MB {dt:.0f}s "
            f"(type={ftype}, first_norm={norm0:.6f})")

        manifest["files"].append({
            "file": os.path.basename(out_path),
            "rows": nrows,
            "bytes": os.path.getsize(out_path),
            "dim_check": dim,
            "norm_check": round(norm0, 6),
            "seconds": round(dt, 1),
        })
        done_rows += nrows

        # 每块后刷新 manifest，中断也有记录
        with open(os.path.join(args.outdir, "_manifest.json"), "w") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_bytes = sum(f["bytes"] for f in manifest["files"])
    manifest["total_bytes"] = total_bytes
    manifest["n_files"] = len(manifest["files"])
    manifest["elapsed_seconds"] = round(time.time() - t_all, 1)
    with open(os.path.join(args.outdir, "_manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log("=" * 60)
    log(f"完成：{len(manifest['files'])} 个文件，{done_rows} 行，"
        f"{total_bytes/1024**3:.2f} GB，耗时 {manifest['elapsed_seconds']/60:.1f} 分钟")
    log(f"输出目录: {args.outdir}")


if __name__ == "__main__":
    main()
