import re

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StopWordsRemover

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


def print_terms(pd_terms, sizes, lens=None, title=""):
    print(f"\n{title}", flush=True)
    for cl in sorted(pd_terms["cluster"].unique()):
        s = pd_terms[pd_terms["cluster"] == cl].sort_values("rk")
        ts = ", ".join(f"{r.w}({r.score:.3f})" for r in s.itertuples())
        extra = ""
        if lens and cl in lens:
            extra = f", len {lens[cl][0]}±{lens[cl][1]}"
        print(f"\n   簇 {cl} (n={sizes.get(cl, '?')}{extra}):", flush=True)
        print(f"      {ts}", flush=True)
