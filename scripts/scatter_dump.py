# 抽取二维散点数据：用前两个主成分作为坐标，配合簇标签。
#
# 全量 194 万点在图上无法辨认结构，按等距分层抽取以保证各簇按比例出现。
# 每个簇抽取上限固定，避免大簇淹没小簇。

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-in", required=True)
    ap.add_argument("--pca-in", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cluster", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("WildChat-Scatter")
             .config("spark.sql.shuffle.partitions", "32")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    lab = spark.read.parquet(args.labels_in).select("item_id", "cluster")
    pca = (spark.read.parquet(args.pca_in)
           .select("item_id", "pca", "lang", "token_len"))

    df = (pca.join(lab, "item_id")
             .withColumn("pc1", F.col("pca").getItem(0))
             .withColumn("pc2", F.col("pca").getItem(1))
             .select("item_id", "cluster", "pc1", "pc2", "lang", "token_len"))

    w = Window.partitionBy("cluster").orderBy(F.rand(args.seed))
    samp = (df.withColumn("rk", F.row_number().over(w))
              .filter(F.col("rk") <= args.per_cluster)
              .drop("rk"))

    n = samp.count()
    print(f"抽样点数 {n:,}", flush=True)
    samp.groupBy("cluster").count().orderBy("cluster").show(20, truncate=False)

    print("各簇前两个主成分的范围", flush=True)
    (samp.groupBy("cluster")
         .agg(F.round(F.min("pc1"), 2).alias("pc1_min"),
              F.round(F.max("pc1"), 2).alias("pc1_max"),
              F.round(F.min("pc2"), 2).alias("pc2_min"),
              F.round(F.max("pc2"), 2).alias("pc2_max"),
              F.round(F.avg("pc1"), 3).alias("pc1_mean"),
              F.round(F.avg("pc2"), 3).alias("pc2_mean"))
         .orderBy("cluster")
         .show(20, truncate=False))

    samp.write.mode("overwrite").option("header", "true").csv(args.out)
    print(f"-> {args.out}", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
