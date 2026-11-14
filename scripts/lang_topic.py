# 各语言在七个簇上的分布。
#
# 用轮级 lang 而非对话级 conv_lang（见 数据处理方法.md §1.2）。
# 短文本的语言检测不可靠，因此同时输出最小样本量阈值下的保留语言列表。

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-in", required=True)
    ap.add_argument("--pca-in", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=2000)
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("WildChat-Lang-Topic")
             .config("spark.sql.shuffle.partitions", "32")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    lab = spark.read.parquet(args.labels_in).select("item_id", "cluster", "token_len")
    meta = spark.read.parquet(args.pca_in).select("item_id", "lang")
    df = lab.join(meta, "item_id").cache()
    n = df.count()

    lang_tot = (df.groupBy("lang").count()
                  .withColumnRenamed("count", "lang_total")
                  .orderBy(F.desc("lang_total")))
    print("=" * 78, flush=True)
    print(f"总行数 {n:,}，语言种类 {lang_tot.count()}", flush=True)
    print("\n样本量前 20 的语言", flush=True)
    lang_tot.show(20, truncate=False)

    keep = [r["lang"] for r in
            lang_tot.filter(F.col("lang_total") >= args.min_n).collect()]
    print(f"样本量 >= {args.min_n} 的语言共 {len(keep)} 种", flush=True)
    print(f"  {keep}", flush=True)

    print("=" * 78, flush=True)
    print("各语言内、七个簇的占比（只列样本量达标的语言）", flush=True)
    cnt = (df.filter(F.col("lang").isin(keep))
             .groupBy("lang", "cluster").count())
    tot = cnt.groupBy("lang").agg(F.sum("count").alias("l_total"))
    share = (cnt.join(tot, "lang")
                .withColumn("share", F.round(F.col("count") / F.col("l_total"), 4))
                .select("lang", "cluster", "count", "share"))
    piv = (share.groupBy("lang")
                .pivot("cluster")
                .agg(F.first("share"))
                .join(lang_tot, "lang")
                .orderBy(F.desc("lang_total")))
    piv.show(30, truncate=False)

    print("=" * 78, flush=True)
    print("各语言最偏好的簇（占比最高）与最回避的簇", flush=True)
    wmax = Window.partitionBy("lang").orderBy(F.desc("share"))
    wmin = Window.partitionBy("lang").orderBy(F.asc("share"))
    top1 = (share.withColumn("rk", F.row_number().over(wmax))
                 .filter(F.col("rk") == 1)
                 .select("lang", F.col("cluster").alias("top_cluster"),
                         F.col("share").alias("top_share")))
    bot1 = (share.withColumn("rk", F.row_number().over(wmin))
                 .filter(F.col("rk") == 1)
                 .select("lang", F.col("cluster").alias("bot_cluster"),
                         F.col("share").alias("bot_share")))
    (top1.join(bot1, "lang").join(lang_tot, "lang")
         .orderBy(F.desc("lang_total"))
         .show(30, truncate=False))

    print("=" * 78, flush=True)
    print("各簇的语言构成（前 5）", flush=True)
    wc = Window.partitionBy("cluster").orderBy(F.desc("n"))
    (df.groupBy("cluster", "lang").count()
       .withColumn("n", F.col("count"))
       .withColumn("rk", F.row_number().over(wc))
       .filter(F.col("rk") <= 5)
       .orderBy("cluster", "rk")
       .select("cluster", "lang", "n")
       .show(60, truncate=False))

    print("=" * 78, flush=True)
    print("各簇的语言熵（越高表示语言构成越分散）", flush=True)
    ent = (df.groupBy("cluster", "lang").count()
             .withColumn("p", F.col("count") / F.sum("count").over(
                 Window.partitionBy("cluster")))
             .groupBy("cluster")
             .agg(F.round(-F.sum(F.col("p") * F.log2(F.col("p"))), 3).alias("entropy"),
                  F.countDistinct("lang").alias("n_langs"),
                  F.sum("count").alias("total"))
             .orderBy("cluster"))
    ent.show(20, truncate=False)

    share.write.mode("overwrite").parquet(args.out)
    print(f"明细 -> {args.out}", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
