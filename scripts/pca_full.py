#!/usr/bin/env python3
"""
全量 PCA：1,943,027 × 2560 -> 64 维

内存要点（这是全量与 10 万条抽样的关键差别）：
  194 万行 × 2560 维在 JVM 里是 39.8 GB（Vector 用 double）。
  因此：
    1. **绝不 cache 2560 维向量** —— 缓存必然 OOM
    2. 不做 repartition（会 shuffle 39.8 GB）—— 保留按文件切分的分区
    3. 把 spark.sql.files.maxPartitionBytes 调小到 64 MB，让每个 task
       只处理约 6,400 行（约 131 MB 向量），8 个并发 task 合计约 1 GB
    4. Spark 的 PCA 用 computeGramianMatrix 的 treeAggregate 实现，
       逐分区累加 2560×2560 的 Gram 矩阵（52 MB），不需要整体常驻

产物：
  hdfs:///wildchat/dwd/pca64_full/        变换后的 64 维向量与元信息
  hdfs:///wildchat/dwd/pca64_full_model/  PCAModel（含投影矩阵与解释方差）

用法（master 容器内）：
  spark-submit --master yarn --deploy-mode client \
      --num-executors 2 --executor-memory 6g --executor-cores 4 \
      --conf spark.sql.files.maxPartitionBytes=64m \
      pca_full.py --input hdfs:///wildchat/ods/vectors_full \
      --out hdfs:///wildchat/dwd/pca64_full \
      --model-out hdfs:///wildchat/dwd/pca64_full_model \
      --k 64
"""
import time
import argparse

import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import PCA
from pyspark.ml.functions import array_to_vector


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--out-partitions", type=int, default=16)
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("WildChat-Full-PCA")
             .config("spark.sql.files.maxPartitionBytes", "64m")
             .config("spark.sql.shuffle.partitions", "64")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    log("1. 读取 parquet")
    df = spark.read.parquet(args.input)
    n = df.count()
    log(f"   行数 {n:,}")

    log("2. array<float> -> VectorUDT（Arrow 批量转换，不 cache）")
    t0 = time.time()
    df = df.withColumn("features", array_to_vector(F.col("embedding"))) \
           .drop("embedding", "content")
    # 用一次 count 触发 action，确认列可读；不 cache
    log(f"   列: {df.columns}")
    log(f"   分区数: {df.rdd.getNumPartitions()}")

    log(f"3. 拟合 PCA(k={args.k}) —— 这一步是主要耗时")
    pca = PCA(k=args.k, inputCol="features", outputCol="pca_features").fit(df)
    ev = np.array(pca.explainedVariance.toArray(), dtype=float)
    log(f"   累计解释方差: {ev.sum():.4f}")
    log(f"   前 10 维方差: {[round(float(x), 5) for x in ev[:10]]}")
    log(f"   拟合耗时 {time.time()-t0:.0f}s")

    log("4. 变换并写出")
    t1 = time.time()
    to_arr = F.udf(lambda v: [float(x) for x in v.toArray()], "array<float>")
    out = (pca.transform(df)
              .withColumn("pca", to_arr(F.col("pca_features")))
              .select("item_id", "pca", "lang", "conv_lang", "token_len", "timestamp")
              .coalesce(args.out_partitions))
    out.write.mode("overwrite").parquet(args.out)
    log(f"   写出完成 {args.out}  耗时 {time.time()-t1:.0f}s")

    log("5. 保存 PCA 模型（含投影矩阵与解释方差）")
    pca.write().overwrite().save(args.model_out)
    log(f"   模型已保存 {args.model_out}")

    log("6. 校验输出")
    chk = spark.read.parquet(args.out)
    cnt = chk.count()
    row = chk.select("pca").first()
    log(f"   输出行数: {cnt:,}  (期望 {n:,}) {'OK' if cnt == n else 'MISMATCH'}")
    log(f"   向量维度: {len(row['pca'])}  (期望 {args.k})")
    log(f"   示例前 4 维: {[round(float(x), 5) for x in row['pca'][:4]]}")

    log("完成")
    spark.stop()


if __name__ == "__main__":
    main()
