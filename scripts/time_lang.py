# 时间 × 语言 的交叉统计，用于判断簇 4 萎缩的原因。
#
# 判别逻辑：
#   若中文在当月总提问中的占比同步下降 -> 用户构成变化（中文用户变少了）
#   若中文占比稳定而簇 4 占比下降     -> 行为变化（中文用户的提问方式变了）
#
# 同时输出各语言在簇 4 内部的占比，区分「簇 4 变小」与「簇 4 的中文纯度变化」。

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-in", required=True)
    ap.add_argument("--pca-in", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("WildChat-Time-Lang")
             .config("spark.sql.shuffle.partitions", "32")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    lab = spark.read.parquet(args.labels_in).select("item_id", "cluster", "token_len")
    meta = (spark.read.parquet(args.pca_in)
            .select("item_id", "lang", "timestamp")
            .withColumn("month", F.substring("timestamp", 1, 7)))
    df = lab.join(meta, "item_id").cache()
    print(f"总行数 {df.count():,}", flush=True)

    print("=" * 78, flush=True)
    print("各语言每月占当月总量的比例（前 8 语言）", flush=True)
    month_tot = df.groupBy("month").count().withColumnRenamed("count", "m_total")
    lang_month = df.groupBy("month", "lang").count()
    ls = (lang_month.join(month_tot, "month")
                    .withColumn("share", F.round(F.col("count") / F.col("m_total"), 4))
                    .orderBy("month", F.desc("count")))
    w = Window.partitionBy("month").orderBy(F.desc("count"))
    pivot_lang = (ls.withColumn("rk", F.row_number().over(w))
                    .filter(F.col("rk") <= 8)
                    .groupBy("month")
                    .pivot("lang")
                    .agg(F.first("share"))
                    .orderBy("month"))
    pivot_lang.show(20, truncate=False)

    print("=" * 78, flush=True)
    print("中文内部：各月落在簇 4 的比例（判别行为变化）", flush=True)
    zh = df.filter(F.col("lang") == "Chinese")
    zh_month = zh.groupBy("month").count().withColumnRenamed("count", "zh_total")
    zh_c4 = (zh.filter(F.col("cluster") == 4)
               .groupBy("month").count().withColumnRenamed("count", "zh_c4"))
    (zh_month.join(zh_c4, "month")
             .withColumn("zh_c4_share",
                         F.round(F.col("zh_c4") / F.col("zh_total"), 4))
             .orderBy("month")
             .show(20, truncate=False))

    print("=" * 78, flush=True)
    print("中文内部：各月落在各簇的比例", flush=True)
    zhc = zh.groupBy("month", "cluster").count()
    zt = zhc.groupBy("month").agg(F.sum("count").alias("t"))
    (zhc.join(zt, "month")
        .withColumn("share", F.round(F.col("count") / F.col("t"), 4))
        .groupBy("month").pivot("cluster").agg(F.first("share"))
        .orderBy("month")
        .show(20, truncate=False))

    print("=" * 78, flush=True)
    print("簇 4 内部的语言构成（各月前 5 语言）", flush=True)
    c4 = df.filter(F.col("cluster") == 4)
    c4m = c4.groupBy("month").count().withColumnRenamed("count", "c4_total")
    c4l = c4.groupBy("month", "lang").count()
    w2 = Window.partitionBy("month").orderBy(F.desc("count"))
    (c4l.join(c4m, "month")
        .withColumn("share", F.round(F.col("count") / F.col("c4_total"), 3))
        .withColumn("rk", F.row_number().over(w2))
        .filter(F.col("rk") <= 5)
        .orderBy("month", "rk")
        .select("month", "lang", "count", "share")
        .show(80, truncate=False))

    print("=" * 78, flush=True)
    print("中文内部的长度分布随时间变化", flush=True)
    (zh.groupBy("month")
       .agg(F.count("*").alias("n"),
            F.round(F.avg("token_len"), 1).alias("avg_tok"),
            F.round(F.expr("percentile_approx(token_len, 0.5)"), 1).alias("p50"),
            F.round(F.expr("percentile_approx(token_len, 0.9)"), 1).alias("p90"))
       .orderBy("month")
       .show(20, truncate=False))

    ls.write.mode("overwrite").parquet(args.out)
    print(f"明细 -> {args.out}", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
