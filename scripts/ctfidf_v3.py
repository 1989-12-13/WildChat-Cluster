#!/usr/bin/env python3
"""
c-TF-IDF v3：四项修正

相比之前版本的改动：

1. 分词
   旧：自写正则 [a-z][a-z0-9_']{2,}，把 Schön→sch、español→espa，还丢掉 AI/ML。
   新：Spark RegexTokenizer(pattern=r"[\\p{L}\\p{N}']+")（Java 正则，Unicode 感知），
       中文再用 jieba 分词。

2. 停用词选择
   旧：手写 LANG_MAP，且用对话级 conv_lang（实为数据集顶层 language，
       是各轮检测结果的众数）。
   新：按 **轮级 lang** 字段选择，两份公开发布的词表按覆盖范围分工——
       · Spark MLlib 官方词表（14 种语言）覆盖的，用 Spark 的
       · 其余语言用 Stopwords ISO（58 种，含中文 794 词），
         语言名到 ISO 码的转换用 pycountry
       全程不手写任何停用词，也不手写语言名映射表。

3. 轮廓系数
   用 sklearn.metrics.silhouette_score（标准定义、欧氏距离），在采样子集上计算。
   不使用 Spark 的 ClusteringEvaluator：它在平方欧氏距离空间计算
   (b²-a²)/max(a²,b²)，数值系统性高于标准定义，且抬高幅度随各点距离比的
   离散程度变化，跨 k 排序可能与标准定义不一致。

4. c-TF-IDF 的 TF
   旧：原始 token 计数。后果是超长文档支配关键词——实测某簇 550 条德语
       贡献 128 万 token（占该簇 54%），使 der/die/und 成为关键词。
   新：逐文档 L2 归一化后再聚合（等价于 sklearn TfidfVectorizer 的 norm='l2'），
       IDF 用 sklearn 的平滑公式 log((1+K)/(1+df)) + 1。

用法（master 容器内）：
  spark-submit --master yarn --deploy-mode client \\
      --num-executors 2 --executor-memory 6g --executor-cores 4 \\
      ctfidf_v3.py --input hdfs:///wildchat/ods/vectors_sample100k \\
      --out hdfs:///wildchat/dws/ctfidf_v3 --k 10
"""
import re
import argparse

import numpy as np
from sklearn.metrics import silhouette_score

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.window import Window
from pyspark.ml.feature import RegexTokenizer, PCA, StopWordsRemover
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import array_to_vector

# ---- Spark 官方停用词表（14 种语言）----
SPARK_LANGS = [
    "danish", "dutch", "english", "finnish", "french", "german",
    "hungarian", "italian", "norwegian", "portuguese", "russian",
    "spanish", "swedish", "turkish",
]
# 数据集里对同一语言使用了「书面挪威语 / 新挪威语」两个标签，
# 而 Spark 只有 norwegian，故仅保留这两个别名，不是通用映射表。
ALIAS = {"bokmal": "norwegian", "nynorsk": "norwegian"}

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

# jieba 在每个 worker 进程内懒加载一次。
# 注意：不能在闭包里捕获 jieba.lcut —— 它是 Tokenizer 实例的绑定方法，
# 无法被 pickle 发送到 executor，会导致 UDF 直接失败。
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
    """构造 {语言名小写: 停用词集合}。

    分工：
      1. Spark MLlib 官方词表覆盖的 14 种语言 -> 用 Spark 的（较保守，
         英文仅 181 词，不会误删 number/value/text 这类关键词）
      2. 其余语言 -> Stopwords ISO（58 种，含中文 794 词）
         语言名到 ISO 639-1 码的转换用 pycountry，避免手写映射表
      3. 两边都没有的语言 -> 不建键，即不做停用词过滤

    lang_names 来自数据本身（`df.select("lang").distinct()`），不硬编码。
    """
    sw = {k: set(StopWordsRemover.loadDefaultStopWords(k)) for k in SPARK_LANGS}

    try:
        import pycountry
        import stopwordsiso
    except ImportError:
        print("   [warn] pycountry / stopwordsiso 不可用，仅使用 Spark 官方词表",
              flush=True)
        return sw

    added = []
    for name in lang_names:
        if not name:
            continue
        key = name.strip().lower()
        if key in sw:                    # Spark 已覆盖
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
    """把 RegexTokenizer 的输出加工成最终 token 列表。

    - 含中日韩字符的 token -> jieba 分词
    - 其余 token -> 小写
    - 按该轮的语言套用停用词表
    - 拉丁/西里尔单字符丢弃（单字母无信息量）
    - 中日韩不做长度过滤：单字功能词由中文停用词表负责，
      单字实词（水、爱、手）应当保留

    闭包只捕获 stopwords（可 pickle 的普通 dict）。
    """

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


def ctfidf_normalized(labeled, k_clusters, topn):
    """c-TF-IDF：逐文档 L2 归一化的 TF + sklearn 平滑 IDF。"""
    # 每篇文档内每个词的计数
    doc = (labeled.select("item_id", "cluster",
                          F.explode(F.col("tokens")).alias("w"))
           .groupBy("item_id", "cluster", "w").count())

    # 逐文档 L2 范数
    norms = (doc.groupBy("item_id")
                .agg(F.sqrt(F.sum(F.col("count") * F.col("count"))).alias("norm")))
    normed = doc.join(norms, "item_id").withColumn("nw", F.col("count") / F.col("norm"))

    # 簇内聚合
    clu = normed.groupBy("cluster", "w").agg(F.sum("nw").alias("tf_raw"))
    tot = clu.groupBy("cluster").agg(F.sum("tf_raw").alias("tot"))
    tf = clu.join(tot, "cluster").withColumn("tf", F.col("tf_raw") / F.col("tot"))

    # 词在多少个簇中出现
    dfreq = clu.groupBy("w").agg(F.countDistinct("cluster").alias("df"))

    # sklearn 平滑 IDF: log((1+N)/(1+df)) + 1
    out = (tf.join(dfreq, "w")
             .withColumn("idf", F.log((1.0 + k_clusters) / (1.0 + F.col("df"))) + 1.0)
             .withColumn("score", F.col("tf") * F.col("idf")))

    w = Window.partitionBy("cluster").orderBy(F.desc("score"))
    return (out.withColumn("rk", F.row_number().over(w))
               .filter(F.col("rk") <= topn)
               .select("cluster", "rk", "w", "score").toPandas())


def print_terms(pd_terms, sizes, lens=None, title=""):
    print(f"\n{title}", flush=True)
    for cl in sorted(pd_terms["cluster"].unique()):
        s = pd_terms[pd_terms["cluster"] == cl].sort_values("rk")
        ts = ", ".join(f"{r.w}({r.score:.2f})" for r in s.itertuples())
        extra = ""
        if lens and cl in lens:
            extra = f", len {lens[cl][0]}±{lens[cl][1]}"
        print(f"\n   簇 {cl} (n={sizes.get(cl, '?')}{extra}):", flush=True)
        print(f"      {ts}", flush=True)


def run_cluster(spark, df, args, tag):
    """对一个 DataFrame 做 PCA -> KMeans -> 轮廓系数 -> c-TF-IDF。"""
    d = df.repartition(args.partitions)
    pca = PCA(k=args.pca_k, inputCol="features", outputCol="pca_features").fit(d)
    ev = float(np.array(pca.explainedVariance.toArray()).sum())
    pdf = pca.transform(d).select(
        "item_id", "tokens", "pca_features", "token_len").cache()
    n = pdf.count()

    km = KMeans(featuresCol="pca_features", predictionCol="cluster",
                k=args.k, seed=args.seed, maxIter=100).fit(pdf)
    lab = km.transform(pdf).cache()
    lab.count()
    print(f"   [{tag}] n={n:,}  PCA方差={ev:.4f}  WSSSE={float(km.summary.trainingCost):,.2f}",
          flush=True)

    # 轮廓系数：sklearn 的标准定义（欧氏距离）。
    # Spark 的 ClusteringEvaluator 在平方欧氏距离空间计算 (b²-a²)/max(a²,b²)，
    # 数值系统性高于标准定义 (b-a)/max(a,b)，且抬高幅度随各点距离比的离散程度变化，
    # 跨 k 的平均值排序可能与标准定义不一致。因此对外报 sklearn 的值，
    # ClusteringEvaluator 仅作过程参考。
    frac = min(1.0, args.sil_sample / n)
    samp = lab.select("pca_features", "cluster").sample(frac, seed=args.seed) \
              .limit(args.sil_sample)
    rows = samp.collect()
    X = np.array([r["pca_features"].toArray() for r in rows], dtype=np.float64)
    y = np.array([r["cluster"] for r in rows], dtype=np.int32)
    sil = float(silhouette_score(X, y, metric="euclidean"))
    print(f"   [{tag}] 轮廓系数 = {sil:.4f}  (sklearn, 标准定义, "
          f"{len(y)} 点采样)", flush=True)

    terms = ctfidf_normalized(lab, args.k, args.topn)
    sizes = {r["cluster"]: r["count"] for r in lab.groupBy("cluster").count().collect()}
    lens = {r["cluster"]: (round(r["avg_tok"], 1), round(r["std_tok"], 1))
            for r in lab.groupBy("cluster")
                      .agg(F.avg("token_len").alias("avg_tok"),
                           F.stddev("token_len").alias("std_tok")).collect()}
    print_terms(terms, sizes, lens, title=f"【{tag} c-TF-IDF（L2 归一化 TF）】")

    lab.unpersist()
    pdf.unpersist()
    return terms, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca-k", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--topn", type=int, default=12)
    ap.add_argument("--langs", default="English,Chinese,Russian")
    ap.add_argument("--sil-sample", type=int, default=10000)
    ap.add_argument("--partitions", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("WildChat-CTFIDF-v3")
             .config("spark.sql.shuffle.partitions", str(args.partitions))
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 74, flush=True)
    print("1. 读取数据，收集实际出现的语言名", flush=True)
    raw = spark.read.parquet(args.input)
    # 只扫 lang 列（parquet 列裁剪），不读向量
    lang_names = sorted(r[0] for r in raw.select("lang").distinct().collect() if r[0])
    print(f"   语言种类: {len(lang_names)}", flush=True)

    stopwords = build_stopwords(lang_names)
    print(f"   停用词表覆盖 {len(stopwords)} 个键，"
          f"未覆盖的语言（不做过滤）: "
          f"{[n for n in lang_names if ALIAS.get(n.lower(), n.lower()) not in stopwords][:8]}",
          flush=True)
    post = make_token_postprocess(stopwords)
    POST = F.udf(lambda ts, lg: post(ts, lg), ArrayType(StringType()))

    print("=" * 74, flush=True)
    print("2. Unicode 分词（RegexTokenizer \\p{L}）+ jieba 中文分词", flush=True)
    base = (raw.withColumn("features", array_to_vector(F.col("embedding")))
               .drop("embedding"))
    rt = RegexTokenizer(inputCol="content", outputCol="raw_tokens",
                        pattern=r"[\p{L}\p{N}']+", gaps=False)
    base = rt.transform(base)
    # 停用词按轮级 lang 选择
    base = base.withColumn("tokens", POST(F.col("raw_tokens"), F.col("lang")))
    base = base.cache()
    print(f"   行数: {base.count():,}", flush=True)

    print("\n   分词示例:", flush=True)
    for r in base.select("lang", "content", "tokens").limit(4).collect():
        print(f"     [{r['lang']}] {(r['content'] or '')[:60]!r}", flush=True)
        print(f"       -> {r['tokens'][:14]}", flush=True)

    all_terms = []
    t, _ = run_cluster(spark, base, args, "全局")
    t = t.assign(scope="global")
    all_terms.append(t)

    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        print("=" * 74, flush=True)
        print(f"分语言: {lang}（按轮级 lang 过滤）", flush=True)
        sub = base.filter(F.col("lang") == lang)
        n = sub.count()
        if n < 500:
            print("   跳过", flush=True)
            continue
        t2, _ = run_cluster(spark, sub, args, lang)
        all_terms.append(t2.assign(scope=lang))

    import pandas as pd
    pd.concat(all_terms).to_csv("/tmp/v3_terms.csv", index=False, encoding="utf-8")
    print("=" * 74, flush=True)
    print("完成", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
