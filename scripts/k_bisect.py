# 在 10 万条抽样上用二分法搜索最优 k 的范围。
#
# 思路：把「最优 k」看作关于 k 的函数 f(k) = 轮廓系数 的极大点。
# 用离散导数符号做区间二分：比较 f(mid) 与 f(mid+1)，
# 上升则最优点在右半区，否则在左半区，每轮区间减半。
#
# 轮廓系数在 k 上不保证严格单峰，因此二分结束后在极值点附近
# 再做一次局部扫描，用「不低于峰值 95% 的连续区间」给出 k 的范围。
#
# 全程使用同一份固定的 10,000 点采样，保证不同 k 之间可比。

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
    ap.add_argument("--lo", type=int, default=2)
    ap.add_argument("--hi", type=int, default=64)
    ap.add_argument("--sil-sample", type=int, default=10000)
    ap.add_argument("--local-span", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--partitions", type=int, default=48)
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("WildChat-K-Bisect")
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

    # 固定采样：所有 k 共用同一批点，数值之间才可比
    frac = min(1.0, args.sil_sample / n)
    samp = pdf.select("pca_features").sample(frac, seed=args.seed) \
              .limit(args.sil_sample).cache()
    m = samp.count()
    X = np.array([r["pca_features"].toArray() for r in samp.collect()],
                 dtype=np.float64)
    print(f"  轮廓系数采样 {m:,} 点（固定，所有 k 共用）", flush=True)

    cache = {}

    def f(k):
        if k in cache:
            return cache[k]
        model = KMeans(featuresCol="pca_features", predictionCol="cluster",
                       k=k, seed=args.seed, maxIter=100).fit(pdf)
        lab = model.transform(samp).select("cluster").collect()
        y = np.array([r["cluster"] for r in lab], dtype=np.int32)
        s = float(silhouette_score(X, y, metric="euclidean"))
        cache[k] = s
        print(f"    k={k:>3}  轮廓系数={s:.6f}", flush=True)
        return s

    print("=" * 60, flush=True)
    print(f"二分搜索 k ∈ [{args.lo}, {args.hi}]", flush=True)

    lo, hi = args.lo, args.hi
    while hi - lo > 2:
        mid = (lo + hi) // 2
        print(f"  区间 [{lo}, {hi}]  mid={mid}", flush=True)
        if f(mid + 1) > f(mid):
            lo = mid + 1
        else:
            hi = mid

    print(f"  区间收敛至 [{lo}, {hi}]", flush=True)

    # 候选区间内逐个评估
    for k in range(lo, hi + 1):
        f(k)
    k_peak = max(range(lo, hi + 1), key=lambda k: cache[k])
    print(f"  区间内最优 k={k_peak}", flush=True)

    print("=" * 60, flush=True)
    print(f"局部扫描 k ∈ [{max(2, k_peak - args.local_span)}, "
          f"{k_peak + args.local_span}]", flush=True)
    for k in range(max(2, k_peak - args.local_span), k_peak + args.local_span + 1):
        f(k)

    peak = cache[k_peak]
    thresh = peak * 0.95
    accept = sorted(k for k, v in cache.items() if v >= thresh and v == v)
    # 取包含峰值的最长连续区间
    runs, cur = [], [k_peak]
    for k in sorted(accept):
        if k in cur:
            continue
        if k == cur[-1] + 1 or k == cur[0] - 1:
            cur.append(k)
            cur.sort()
        else:
            runs.append(cur)
            cur = [k]
    runs.append(cur)
    runs.sort(key=len, reverse=True)
    best_run = runs[0]

    print("=" * 60, flush=True)
    print(f"{'k':>4}{'轮廓系数':>14}", flush=True)
    for k in sorted(cache):
        mark = ""
        if k == k_peak:
            mark = "  <- 峰值"
        elif k in best_run:
            mark = "  <- 平台"
        print(f"{k:>4}{cache[k]:>14.6f}{mark}", flush=True)

    print("=" * 60, flush=True)
    print(f"峰值 k = {k_peak}，轮廓系数 = {peak:.6f}", flush=True)
    print(f"阈值 = 峰值的 95% = {thresh:.6f}", flush=True)
    print(f"k 的范围 = [{best_run[0]}, {best_run[-1]}]（共 {len(best_run)} 个取值）",
          flush=True)

    spark.stop()


if __name__ == "__main__":
    main()
