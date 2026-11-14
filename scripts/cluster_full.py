# 全量 1,943,027 条的 KMeans 聚类与 c-TF-IDF。
#
# 输入 pca64_full 只含 64 维向量与元信息，没有 content。
# c-TF-IDF 需要文本，因此从原始 parquet 按列裁剪读取 item_id 与 content，
# 避免把 2560 维向量一起读进来。
#
# 分阶段执行，簇标签在阶段一结束即写入 HDFS，后续阶段失败不必从头重跑。

import re
import argparse

import numpy as np
from sklearn.metrics import silhouette_score

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.window import Window
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import array_to_vector

SPARK_LANGS = [
    "danish", "dutch", "english", "finnish", "french", "german",
    "hungarian", "italian", "norwegian", "portuguese", "russian",
    "spanish", "swedish", "turkish",
]
ALIAS = {"bokmal": "norwegian", "nynorsk": "norwegian"}

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

_JIEBA = None


def _jieba_mod():
    global _JIEBA
    if _JIEBA is None:
        import logging as _lg
        import jieba as _jb
        _jb.setLogLevel(_lg.ERROR)
        _JIEBA = _jb
    return _JIEBA


def build_stopwords(lang_names):
    sw = {k: set(StopWordsRemover.loadDefaultStopWords(k)) for k in SPARK_LANGS}

    import pycountry
    import stopwordsiso

    added = []
    for name in lang_names:
        if not name:
            continue
        key = name.strip().lower()
        if key in sw:
            continue
        code = None
        try:
            lg = pycountry.languages.get(name=name.strip())
            if lg is not None:
                code = getattr(lg, "alpha_2", None)
        except LookupError:
            pass
        if code and stopwordsiso.has_lang(code):
            sw[key] = set(stopwordsiso.stopwords(code))
            added.append((name, code))

    if added:
        print(f"   Stopwords ISO 补充 {len(added)} 种语言: "
              f"{', '.join(f'{n}({c})' for n, c in added[:12])}"
              f"{' ...' if len(added) > 12 else ''}", flush=True)
    return sw


def make_token_postprocess(stopwords):

    def post(tokens, lang):
        if not tokens:
            return []
        jb = _jieba_mod()
        key = ALIAS.get((lang or "").lower(), (lang or "").lower())
        sw = stopwords.get(key, set())
        out = []
        for t in tokens:
            t = t.strip("'").lower()
            if not t:
                continue
            if CJK_RE.search(t):
                for w in jb.lcut(t):
                    w = w.strip()
                    if w and w not in sw:
                        out.append(w)
            else:
                if len(t) >= 2 and t not in sw:
                    out.append(t)
        return out[:2000]

    return post


def ctfidf_top_terms(labeled, k_clusters, topn):
    doc = (labeled.select("item_id", "cluster",
                          F.explode(F.col("tokens")).alias("w"))
           .groupBy("item_id", "cluster", "w").count())

    norms = (doc.groupBy("item_id")
                .agg(F.sqrt(F.sum(F.col("count") * F.col("count"))).alias("norm")))
    normed = doc.join(norms, "item_id").withColumn("nw", F.col("count") / F.col("norm"))

    clu = normed.groupBy("cluster", "w").agg(F.sum("nw").alias("tf_raw"))
    tot = clu.groupBy("cluster").agg(F.sum("tf_raw").alias("tot"))
    tf = clu.join(tot, "cluster").withColumn("tf", F.col("tf_raw") / F.col("tot"))

    dfreq = clu.groupBy("w").agg(F.countDistinct("cluster").alias("df"))

    out = (tf.join(dfreq, "w")
             .withColumn("idf", F.log((1.0 + k_clusters) / (1.0 + F.col("df"))) + 1.0)
             .withColumn("score", F.col("tf") * F.col("idf")))

    w = Window.partitionBy("cluster").orderBy(F.desc("score"))
    return (out.withColumn("rk", F.row_number().over(w))
               .filter(F.col("rk") <= topn)
               .select("cluster", "rk", "w", "score").toPandas())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca-in", required=True)
    ap.add_argument("--raw-in", required=True)
    ap.add_argument("--labels-out", required=True)
    ap.add_argument("--profile-out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--sil-sample", type=int, default=10000)
    ap.add_argument("--partitions", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName(f"WildChat-Full-KMeans-k{args.k}")
             .config("spark.sql.shuffle.partitions", str(args.partitions))
             .config("spark.sql.files.maxPartitionBytes", "128m")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    print("阶段一：读取 64 维向量并聚类", flush=True)
    pca = (spark.read.parquet(args.pca_in)
           .withColumn("pca_features", array_to_vector(F.col("pca")))
           .drop("pca"))
    n = pca.count()
    print(f"  行数 {n:,}", flush=True)

    vec = pca.select("item_id", "pca_features", "lang", "conv_lang",
                     "token_len", "timestamp").repartition(args.partitions)
    vec = vec.cache()
    print(f"  已缓存 {vec.count():,} 行（64 维 double 约 {n * 64 * 8 / 1024 ** 3:.2f} GB）",
          flush=True)

    km = KMeans(featuresCol="pca_features", predictionCol="cluster",
                k=args.k, seed=args.seed, maxIter=100).fit(vec)
    print(f"  WSSSE = {float(km.summary.trainingCost):,.0f}", flush=True)

    labeled = km.transform(vec).select(
        "item_id", "cluster", "lang", "conv_lang", "token_len", "timestamp").cache()
    cnt = labeled.count()
    print(f"  k={args.k} 标注完成，{cnt:,} 行", flush=True)

    sizes = {r["cluster"]: r["count"]
             for r in labeled.groupBy("cluster").count().collect()}
    lens = {r["cluster"]: (round(r["avg_tok"], 1), round(r["std_tok"], 1))
            for r in labeled.groupBy("cluster")
                             .agg(F.avg("token_len").alias("avg_tok"),
                                  F.stddev("token_len").alias("std_tok")).collect()}
    print("\n  各簇规模与长度:", flush=True)
    for c in sorted(sizes):
        a, sd = lens[c]
        print(f"   簇 {c}: n={sizes[c]:>9,}  len {a}±{sd}", flush=True)

    print("\n  各簇语言构成（前 4）:", flush=True)
    lw = Window.partitionBy("cluster").orderBy(F.desc("n"))
    (labeled.groupBy("cluster", "lang").count()
            .withColumn("n", F.col("count"))
            .withColumn("rk", F.row_number().over(lw))
            .filter(F.col("rk") <= 4)
            .orderBy("cluster", "rk")
            .select("cluster", "lang", "n")
            .show(60, truncate=False))

    print("  写出簇标签", flush=True)
    (labeled.select("item_id", "cluster", "token_len")
            .write.mode("overwrite").parquet(args.labels_out))
    print(f"  -> {args.labels_out}", flush=True)

    print("\n阶段二：轮廓系数", flush=True)
    frac = min(1.0, args.sil_sample / n)
    samp = vec.select("pca_features").sample(frac, seed=args.seed) \
              .limit(args.sil_sample).cache()
    sil_lab = km.transform(samp).select("pca_features", "cluster").collect()
    X = np.array([r["pca_features"].toArray() for r in sil_lab], dtype=np.float64)
    y = np.array([r["cluster"] for r in sil_lab], dtype=np.int32)
    s = float(silhouette_score(X, y, metric="euclidean"))
    print(f"  轮廓系数 = {s:.6f}（sklearn，标准定义，{len(y):,} 点采样）", flush=True)
    samp.unpersist()

    vec.unpersist()

    print("\n阶段三：读取文本并分词", flush=True)
    raw = spark.read.parquet(args.raw_in).select("item_id", "content")
    lang_names = sorted(r[0] for r in labeled.select("lang").distinct().collect() if r[0])
    print(f"  语言种类 {len(lang_names)}", flush=True)

    stopwords = build_stopwords(lang_names)
    post = make_token_postprocess(stopwords)
    POST = F.udf(lambda ts, lg: post(ts, lg), ArrayType(StringType()))

    rt = RegexTokenizer(inputCol="content", outputCol="raw_tokens",
                        pattern=r"[\p{L}\p{N}']+", gaps=False)
    tok = rt.transform(raw).drop("content")
    joined = labeled.drop("conv_lang", "timestamp").join(tok, "item_id")
    joined = joined.withColumn("tokens", POST(F.col("raw_tokens"), F.col("lang")))
    joined = joined.cache()
    print(f"  分词完成，{joined.count():,} 行", flush=True)

    print("\n阶段四：c-TF-IDF", flush=True)
    terms = ctfidf_top_terms(joined, args.k, args.topn)
    print(f"\n【全量 k={args.k} c-TF-IDF】", flush=True)
    for cl in sorted(terms["cluster"].unique()):
        sub = terms[terms["cluster"] == cl].sort_values("rk")
        ts = ", ".join(f"{r.w}({r.score:.3f})" for r in sub.itertuples())
        a, sd = lens.get(cl, (0, 0))
        print(f"\n   簇 {cl} (n={sizes.get(cl, 0):,}, len {a}±{sd}):", flush=True)
        print(f"      {ts}", flush=True)

    print("\n阶段五：写出簇画像", flush=True)
    prof = (joined.groupBy("cluster", "lang")
                  .agg(F.count("*").alias("n"),
                       F.round(F.avg("token_len"), 1).alias("avg_tok"),
                       F.sum("token_len").alias("sum_tok"))
                  .orderBy("cluster", F.desc("n")))
    prof.write.mode("overwrite").parquet(args.profile_out)
    print(f"  -> {args.profile_out}", flush=True)

    print("完成", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
