import argparse

import numpy as np
from sklearn.metrics import silhouette_score

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import array_to_vector


def evaluate_both(spark, pdf, k, tag, sil_sample=10000, seed=42):
    km = KMeans(featuresCol="pca_features", predictionCol="cluster",
                k=k, seed=seed, maxIter=100).fit(pdf)
    lab = km.transform(pdf).cache()
    n = lab.count()

    frac = min(1.0, sil_sample / n)
    samp = lab.select("pca_features", "cluster").sample(frac, seed=seed) \
              .limit(sil_sample).cache()
    m = samp.count()

    ev = ClusteringEvaluator(featuresCol="pca_features",
                             predictionCol="cluster",
                             metricName="silhouette")
    spark_sil = float(ev.evaluate(samp))

    rows = samp.select("pca_features", "cluster").collect()
    X = np.array([r["pca_features"].toArray() for r in rows], dtype=np.float64)
    y = np.array([r["cluster"] for r in rows], dtype=np.int32)
    sk_sil = float(silhouette_score(X, y, metric="euclidean"))

    # 由平方距离口径反推标准口径：s_sq = 1 - r^2，s_std = 1 - r
    r = np.sqrt(max(0.0, 1.0 - spark_sil))
    back = 1.0 - r

    print(f"{tag:<10}{n:>8}{m:>8}{spark_sil:>14.6f}{sk_sil:>14.6f}"
          f"{spark_sil - sk_sil:>12.6f}{back:>14.6f}", flush=True)

    samp.unpersist()
    lab.unpersist()
    return spark_sil, sk_sil, back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pca-k", type=int, default=64)
    ap.add_argument("--langs", default="English,Chinese,Russian")
    ap.add_argument("--sil-sample", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("WildChat-Silhouette-Compare")
             .config("spark.sql.shuffle.partitions", "24")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    base = (spark.read.parquet(args.input)
            .withColumn("features", array_to_vector(F.col("embedding")))
            .drop("embedding")
            .cache())
    base.count()

    print(f"{'范围':<10}{'总行数':>8}{'采样':>8}"
          f"{'Spark(平方)':>14}{'sklearn(标准)':>14}{'差值':>12}{'反推标准':>14}",
          flush=True)
    print("-" * 80, flush=True)

    for scope in ["全局"] + [x.strip() for x in args.langs.split(",") if x.strip()]:
        sub = base if scope == "全局" else base.filter(F.col("lang") == scope)
        if scope != "全局" and sub.count() < 500:
            continue
        d = sub.repartition(24)
        pca = PCA(k=args.pca_k, inputCol="features", outputCol="pca_features").fit(d)
        pdf = pca.transform(d).select("item_id", "pca_features").cache()
        pdf.count()
        evaluate_both(spark, pdf, args.k, scope, args.sil_sample, args.seed)
        pdf.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()
