import theme  # noqa: F401  统一图表配色
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
DATA = os.path.join(HERE, "..", "out")

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

raw = pd.read_csv(os.path.join(DATA, "scatter_k7.csv"),
                  dtype=str)
# getmerge 会把每个分区的表头一起拼进来，需要剔除
raw = raw[raw["item_id"] != "item_id"]
df = raw.dropna(subset=["cluster", "pc1", "pc2"]).copy()
df["cluster"] = df["cluster"].astype(int)
df["pc1"] = df["pc1"].astype(float)
df["pc2"] = df["pc2"].astype(float)
df["token_len"] = pd.to_numeric(df["token_len"], errors="coerce").fillna(0).astype(int)
print("总点数:", len(df))
print(df.groupby("cluster").size())

names = {
    0: "簇0 跨语言超短指令碎片",
    1: "簇1 同人剧本与角色扮演",
    2: "簇2 代码解释与报错调试",
    3: "簇3 极短多语言问候",
    4: "簇4 超短中文单句提问",
    5: "簇5 Midjourney 提示词模板",
    6: "簇6 学术作业与考试题",
}
cmap = lambda i: theme.PALETTE[i]

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], hspace=0.24, wspace=0.22)

ax = fig.add_subplot(gs[0, :])
for c in range(7):
    sub = df[df.cluster == c]
    ax.scatter(sub.pc1, sub.pc2, s=1.6, alpha=0.30, c=[cmap(c)],
               label=f"{names[c]} ({len(sub):,})", rasterized=True, linewidths=0)
    mx, my = sub.pc1.median(), sub.pc2.median()
    ax.plot(mx, my, "o", ms=13, mfc="white", mec=cmap(c), mew=2.4, zorder=5)
    ax.plot(mx, my, "o", ms=4, c=cmap(c), zorder=6)

ax.set_xlabel("PC1（与 log 长度的相关系数 +0.618）", fontsize=11)
ax.set_ylabel("PC2（相关系数 −0.351）", fontsize=11)
ax.set_title("全量 1,943,027 条：七个簇在 PC1–PC2 平面上的分布（每簇随机 8,000 点，空心圈为各簇中位数）",
             fontsize=12.5, pad=12)
ax.legend(fontsize=9, loc="upper right", framealpha=0.92, markerscale=6)
ax.grid(alpha=0.22, ls="--")
ax.set_xlim(-0.65, 0.75)
ax.set_ylim(-0.55, 0.85)

ax2 = fig.add_subplot(gs[1, 0])
zh = df[df.lang == "Chinese"]
en = df[df.lang == "English"]
ax2.scatter(en.pc1, en.pc2, s=1.5, alpha=0.22, c="#B8AA98", label=f"English ({len(en):,})",
            rasterized=True, linewidths=0)
ax2.scatter(zh.pc1, zh.pc2, s=1.5, alpha=0.30, c=theme.PRIMARY, label=f"Chinese ({len(zh):,})",
            rasterized=True, linewidths=0)
ax2.set_xlabel("PC1", fontsize=10.5)
ax2.set_ylabel("PC2", fontsize=10.5)
ax2.set_title("按语言着色：中文与英文的位置差异", fontsize=11.5, pad=8)
ax2.legend(fontsize=9, loc="upper right", framealpha=0.9, markerscale=6)
ax2.grid(alpha=0.22, ls="--")
ax2.set_xlim(-0.65, 0.75)
ax2.set_ylim(-0.55, 0.85)

ax3 = fig.add_subplot(gs[1, 1])
short = df[df.token_len <= 30]
long_ = df[df.token_len > 300]
mid = df[(df.token_len > 30) & (df.token_len <= 300)]
ax3.scatter(mid.pc1, mid.pc2, s=1.5, alpha=0.18, c="#B8AA98",
            label=f"31–300 tok ({len(mid):,})", rasterized=True, linewidths=0)
ax3.scatter(short.pc1, short.pc2, s=1.5, alpha=0.30, c="#6B8E8A",
            label=f"≤30 tok ({len(short):,})", rasterized=True, linewidths=0)
ax3.scatter(long_.pc1, long_.pc2, s=1.5, alpha=0.30, c=theme.ACCENT,
            label=f">300 tok ({len(long_):,})", rasterized=True, linewidths=0)
ax3.set_xlabel("PC1", fontsize=10.5)
ax3.set_ylabel("PC2", fontsize=10.5)
ax3.set_title("按长度着色：PC1 即长度轴", fontsize=11.5, pad=8)
ax3.legend(fontsize=9, loc="upper right", framealpha=0.9, markerscale=6)
ax3.grid(alpha=0.22, ls="--")
ax3.set_xlim(-0.65, 0.75)
ax3.set_ylim(-0.55, 0.85)

plt.savefig(os.path.join(FIG, "scatter_k7.png"),
            dpi=150, bbox_inches="tight")
print("saved")
