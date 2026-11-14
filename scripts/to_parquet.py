#!/usr/bin/env python3
"""
把 out/vectors.npy + out/meta.parquet 转成 Spark 可读的 parquet。

为什么在进 HDFS 之前转：
  HDFS 只是分布式文件系统，不解析格式；Spark 无法读取 .npy。
  因此必须先用 numpy 读出 npy，写成 parquet，再 put 进 HDFS。

用法：
  # smoke：只取前 100 条
  python3 to_parquet.py --limit 100 --out shared/vectors_smoke.parquet

  # 全量：不传 --limit
  python3 to_parquet.py --out shared/vectors_full.parquet --rows-per-file 50000
"""
import os
import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.environ.get("WILDCHAT_ROOT", ".")
OUT_DIR = os.path.join(ROOT, "out")
VEC = os.path.join(OUT_DIR, "vectors.npy")
META = os.path.join(OUT_DIR, "meta.parquet")

# 与转换有关的元信息列（原文 content 很占空间，全量时可选跳过）
META_COLS = [
    "item_id", "turn_identifier", "conv_turn_idx", "lang", "conv_lang",
    "model", "country", "state", "timestamp", "token_len", "truncated",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="只取前 N 条（smoke 用）")
    ap.add_argument("--out", required=True, help="输出 parquet 路径")
    ap.add_argument("--rows-per-file", type=int, default=None,
                    help="按行数切分为多个文件（全量时建议 50000）")
    ap.add_argument("--include-content", action="store_true",
                    help="是否把原文 content 一并写入（默认不写，体积大）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    # ---- 读向量（mmap，不整体载入） ----
    vec = np.load(VEC, mmap_mode="r")
    n_total = vec.shape[0]
    n = n_total if args.limit is None else min(args.limit, n_total)
    print(f"vectors: {vec.shape} -> 取前 {n} 条")

    # ---- 读元信息 ----
    cols = list(META_COLS) + (["content"] if args.include_content else [])
    meta = pq.read_table(META, columns=cols).slice(0, n)
    assert meta.num_rows == n, f"meta 行数 {meta.num_rows} != {n}"

    # ---- 组装数组列 ----
    block = np.asarray(vec[:n], dtype=np.float32)      # (n, 2560)
    print(f"载入向量块: {block.shape}  {block.nbytes/1024**2:.1f} MB")

    # array<float> 列：list<float>，保持 2560 维不变
    emb_type = pa.list_(pa.float32())
    emb = pa.array(list(block), type=emb_type)

    table = meta.append_column("embedding", emb)

    # ---- 写出 ----
    def write(tbl, path):
        pq.write_table(tbl, path, compression="zstd")
        sz = os.path.getsize(path) / 1024**2
        print(f"  -> {path}  {tbl.num_rows} 行  {sz:.1f} MB")

    if args.rows_per_file and n > args.rows_per_file:
        base = os.path.splitext(args.out)[0]
        nfile = 0
        for start in range(0, n, args.rows_per_file):
            end = min(start + args.rows_per_file, n)
            write(table.slice(start, end - start), f"{base}-{nfile:05d}.parquet")
            nfile += 1
        print(f"共写出 {nfile} 个文件")
    else:
        write(table, args.out)

    # ---- 自检 ----
    chk = pq.read_table(args.out if not args.rows_per_file else
                        f"{os.path.splitext(args.out)[0]}-00000.parquet")
    print("\n=== 自检 ===")
    print("schema:", [f.name for f in chk.schema])
    print(f"rows: {chk.num_rows}")
    if chk.num_rows > 0:
        e = chk.column("embedding")[0].as_py()
        print(f"embedding[0] 维度: {len(e)}  dtype: {type(e[0]).__name__}")
        print(f"embedding[0][:4]: {[round(x,5) for x in e[:4]]}")
        v = np.array(e, dtype=np.float32)
        print(f"L2 范数: {np.linalg.norm(v):.6f}  (应为 1.0)")


if __name__ == "__main__":
    main()
