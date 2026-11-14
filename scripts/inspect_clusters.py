# 抽样检查全量 k=8 中语义可疑的簇。
#
# 关注两点：
#   1. 簇 0 与簇 5 的关键词高度重叠（give / write / one / make / please），
#      需要确认它们是两类不同的请求，还是被多切了一刀。
#   2. 簇 7 中出现 9,682 条 Maori，该语言使用量极小，需确认是否为检测误判。

import argparse
import random

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def sample_cluster(labeled, raw, cluster, n, seed):
    ids = (labeled.filter(F.col("cluster") == cluster)
                  .select("item_id", "token_len", "lang")
                  .orderBy(F.rand(seed))
                  .limit(n))
    return ids.join(raw, "item_id").select("token_len", "content", "lang").collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-in", required=True)
    ap.add_argument("--pca-in", required=True)
    ap.add_argument("--raw-in", required=True)
    ap.add_argument("--clusters", default="0,5")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--maori-cluster", type=int, default=7)
    ap.add_argument("--maori-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("WildChat-Cluster-Inspect")
             .config("spark.sql.shuffle.partitions", "64")
             .config("spark.sql.files.maxPartitionBytes", "128m")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # 标签文件只含 item_id / cluster / token_len，lang 需从 pca64_full 取
    labeled = (spark.read.parquet(args.labels_in)
               .join(spark.read.parquet(args.pca_in).select("item_id", "lang"),
                     "item_id"))
    raw = spark.read.parquet(args.raw_in).select("item_id", "content")
    raw = raw.cache()

    for cl in [int(x) for x in args.clusters.split(",") if x.strip()]:
        print("=" * 78, flush=True)
        print(f"簇 {cl}", flush=True)
        rows = sample_cluster(labeled, raw, cl,
                              args.samples, args.seed + cl * 1000)
        for r in rows:
            txt = (r["content"] or "").replace("\n", " ")[:150]
            print(f"  [{r['lang']:<10} {r['token_len']:>5}tok] {txt}", flush=True)

    print("=" * 78, flush=True)
    print(f"簇 {args.maori_cluster} 中标为 Maori 的样本", flush=True)
    ids = (labeled.filter((F.col("cluster") == args.maori_cluster)
                          & (F.col("lang") == "Maori"))
                  .select("item_id")
                  .orderBy(F.rand(args.seed))
                  .limit(args.maori_samples))
    rows = (ids.join(labeled, "item_id")
               .join(raw, "item_id")
               .select("token_len", "content").collect())
    for r in rows:
        txt = (r["content"] or "").replace("\n", " ")[:150]
        print(f"  [{r['token_len']:>5}tok] {txt}", flush=True)

    print("=" * 78, flush=True)
    print("对照：同样的 sample 在簇 0 与簇 5 中的 lang 分布", flush=True)
    for cl in [0, 5]:
        print(f"\n  簇 {cl}:", flush=True)
        (labeled.filter(F.col("cluster") == cl)
                .groupBy("lang").count()
                .orderBy(F.desc("count"))
                .show(8, truncate=False))

    spark.stop()


if __name__ == "__main__":
    main()
