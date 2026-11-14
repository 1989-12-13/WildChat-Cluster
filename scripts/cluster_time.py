# 各簇在时间上的分布。
#
# 输入：k=7 的簇标签 + pca64_full 中的 timestamp。
# 输出：按月 × 簇 的计数与占比，以及各簇的生命周期指标。
#
# 占比单独计算，因为每月总量差异很大，绝对数量的变化会掩盖结构变化。

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-in", required=True)
    ap.add_argument("--pca-in", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=7)
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("WildChat-Cluster-Time")
             .config("spark.sql.shuffle.partitions", "32")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    lab = spark.read.parquet(args.labels_in).select("item_id", "cluster", "token_len")
    meta = (spark.read.parquet(args.pca_in)
            .select("item_id", "timestamp", "lang")
            .withColumn("month", F.substring("timestamp", 1, 7)))

    df = lab.join(meta, "item_id").cache()
    n = df.count()
    months = df.select("month").distinct().count()
    print(f"行数 {n:,}  月份数 {months}", flush=True)

    print("=" * 70, flush=True)
    print("每月总量", flush=True)
    (df.groupBy("month").count().orderBy("month").show(30, truncate=False))

    print("=" * 70, flush=True)
    print("month × cluster 计数", flush=True)
    cnt = (df.groupBy("month", "cluster").count()
             .orderBy("month", "cluster"))
    cnt.show(120, truncate=False)

    print("=" * 70, flush=True)
    print("month × cluster 占该月比例（保留 4 位小数）", flush=True)
    month_tot = df.groupBy("month").count().withColumnRenamed("count", "m_total")
    share = (cnt.join(month_tot, "month")
                .withColumn("share", F.round(F.col("count") / F.col("m_total"), 4))
                .select("month", "cluster", "count", "share")
                .orderBy("month", "cluster"))
    share.show(120, truncate=False)

    print("=" * 70, flush=True)
    print("各簇月度占比：透视（行=月，列=簇）", flush=True)
    piv = (share.groupBy("month")
                .pivot("cluster")
                .agg(F.first("share"))
                .orderBy("month"))
    piv.show(30, truncate=False)

    print("=" * 70, flush=True)
    print("各簇生命周期", flush=True)
    life = (df.groupBy("cluster")
              .agg(F.min("month").alias("first_month"),
                   F.max("month").alias("last_month"),
                   F.count("*").alias("total"),
                   F.round(F.avg("token_len"), 1).alias("avg_tok"))
              .orderBy("cluster"))
    life.show(20, truncate=False)

    print("=" * 70, flush=True)
    print("各簇占比的变化幅度（最大月占比 - 最小月占比）", flush=True)
    sw = Window.partitionBy("cluster")
    amp = (share.withColumn("max_share", F.max("share").over(sw))
                .withColumn("min_share", F.min("share").over(sw))
                .select("cluster", "max_share", "min_share")
                .distinct()
                .withColumn("amplitude",
                            F.round(F.col("max_share") - F.col("min_share"), 4))
                .orderBy(F.desc("amplitude")))
    amp.show(20, truncate=False)

    share.write.mode("overwrite").parquet(args.out)
    print(f"占比明细 -> {args.out}", flush=True)

    spark.stop()


if __name__ == "__main__":
    main()
