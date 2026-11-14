# 高位区逐点扫描：k = 2..16，步长 1。
#
# 已有步长 2 的扫描显示 k<=28 是缓降平台、峰值在 k=4..8。
# 本脚本在该区间以步长 1 加密，确定峰值的确切位置与其邻域形状。
#
# 所有 k 共用同一份固定采样点，保证数值之间可比。

import argparse

import numpy as np
from sklearn.metrics import silhouette_score

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import array_to_vector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--pca-k", type=int, default=64)
    ap.add_argument("--start", type=int, default=2)
    ap.add_argument("--stop", type=int, default=16)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--sil-sample", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--partitions", type=int, default=48)
    args = ap.parse_args()

    ks = list(range(args.start, args.stop + 1, args.step))

    spark = (SparkSession.builder
             .appName("WildChat-K-FineScan")
             .config("spark.sql.shuffle.partitions", str(args.partitions))
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    print("读取数据并做 PCA", flush=True)
    base = (spark.read.parquet(args.input)
            .withColumn("features", array_to_vector(F.col("embedding")))
            .drop("embedding"))
    n = base.count()
    print(f"  行数 {n:,}", flush=True)

    d = base.repartition(args.partitions)
    pca = PCA(k=args.pca_k, inputCol="features", outputCol="pca_features").fit(d)
    ev = float(np.array(pca.explainedVariance.toArray()).sum())
    pdf = pca.transform(d).select("item_id", "pca_features").cache()
    pdf.count()
    print(f"  PCA({args.pca_k}) 累计解释方差 {ev:.4f}，已缓存", flush=True)

    frac = min(1.0, args.sil_sample / n)
    samp = pdf.select("pca_features").sample(frac, seed=args.seed) \
              .limit(args.sil_sample).cache()
    X = np.array([r["pca_features"].toArray() for r in samp.collect()],
                 dtype=np.float64)
    print(f"  轮廓系数采样 {len(X):,} 点（固定，所有 k 共用）", flush=True)

    print("=" * 52, flush=True)
    print(f"扫描 k = {ks}（步长 {args.step}）", flush=True)

    results = {}
    for k in ks:
        model = KMeans(featuresCol="pca_features", predictionCol="cluster",
                       k=k, seed=args.seed, maxIter=100).fit(pdf)
        wssse = float(model.summary.trainingCost)
        y = np.array([r["cluster"] for r in
                      model.transform(samp).select("cluster").collect()],
                     dtype=np.int32)
        s = float(silhouette_score(X, y, metric="euclidean"))
        results[k] = (s, wssse)
        print(f"  k={k:>3}  轮廓系数={s:.6f}  WSSSE={wssse:,.0f}", flush=True)

    vals = np.array([results[k][0] for k in ks])
    diffs = np.diff(vals)

    print("=" * 52, flush=True)
    print(f"{'k':>4}{'轮廓系数':>13}{'差分':>13}", flush=True)
    for i, k in enumerate(ks):
        dv = "" if i == 0 else f"{diffs[i-1]:+.6f}"
        print(f"{k:>4}{vals[i]:>13.6f}{dv:>13}", flush=True)

    best_i = int(np.argmax(vals))
    print("=" * 52, flush=True)
    print(f"峰值 k={ks[best_i]}  轮廓系数={vals[best_i]:.6f}", flush=True)
    print(f"极差 {vals.max() - vals.min():.6f}", flush=True)
    print(f"相邻最大跳变 {np.abs(diffs).max():.6f}", flush=True)
    if 0 < best_i < len(ks) - 1:
        print(f"峰值两侧落差 左 {vals[best_i] - vals[best_i-1]:+.6f}  "
              f"右 {vals[best_i] - vals[best_i+1]:+.6f}", flush=True)

    spark.stop()


if __name__ == "__main__":
    main()
