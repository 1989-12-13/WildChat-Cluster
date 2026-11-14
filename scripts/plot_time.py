import theme  # noqa: F401  统一图表配色
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

months = ["2023-04", "2023-05", "2023-06", "2023-07", "2023-08", "2023-09",
          "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03",
          "2024-04"]
x = np.arange(len(months))

shares = {
    0: [0.2323, 0.2443, 0.2761, 0.2819, 0.2382, 0.2656, 0.2281, 0.2645, 0.2637, 0.2649, 0.3016, 0.2958, 0.2612],
    1: [0.0880, 0.1063, 0.1233, 0.1501, 0.1395, 0.1190, 0.1232, 0.0911, 0.1227, 0.1582, 0.1587, 0.1288, 0.1312],
    2: [0.2401, 0.2053, 0.1897, 0.1733, 0.1710, 0.1514, 0.1423, 0.1691, 0.1852, 0.1764, 0.1657, 0.1742, 0.1844],
    3: [0.0164, 0.0164, 0.0162, 0.0186, 0.0170, 0.0157, 0.0145, 0.0181, 0.0178, 0.0200, 0.0160, 0.0151, 0.0108],
    4: [0.2316, 0.1740, 0.1572, 0.1378, 0.1474, 0.1081, 0.0840, 0.1029, 0.0749, 0.0436, 0.0240, 0.0371, 0.0689],
    5: [0.0014, 0.0007, 0.0003, 0.0445, 0.0704, 0.0676, 0.0487, 0.0460, 0.0559, 0.0674, 0.0619, 0.0940, 0.0939],
    6: [0.1902, 0.2531, 0.2372, 0.1938, 0.2165, 0.2727, 0.3592, 0.3081, 0.2799, 0.2695, 0.2721, 0.2550, 0.2497],
}
names = {
    0: "簇0 跨语言超短指令碎片",
    1: "簇1 同人剧本与角色扮演",
    2: "簇2 代码解释与报错调试",
    3: "簇3 极短多语言问候",
    4: "簇4 超短中文单句提问",
    5: "簇5 Midjourney 提示词模板",
    6: "簇6 学术作业与考试题",
}
totals = [146291, 273123, 254848, 167685, 118719, 95015, 114255,
          172503, 138526, 73571, 88204, 145031, 155256]

fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.2], hspace=0.28, wspace=0.16)

ax_vol = fig.add_subplot(gs[0, :])
ax_vol.bar(x, np.array(totals) / 1000, color="#C4B8A8", width=0.6)
ax_vol.set_ylabel("每月提问量（千）", fontsize=11)
ax_vol.set_title("WildChat 全量 1,943,027 条 user 轮次：总量与七个簇的占比变化",
                 fontsize=13, pad=12)
ax_vol.grid(alpha=0.3, ls="--", axis="y")
ax_vol.set_xticks(x)
ax_vol.set_xticklabels([])
ax_vol.set_ylim(0, 300)
for i, v in enumerate(totals):
    if i in (1, 9):
        ax_vol.text(i, v / 1000 + 6, f"{v/1000:.0f}k", ha="center", fontsize=9)

cmap = lambda i: theme.PALETTE[i]

ax1 = fig.add_subplot(gs[1, 0])
for c in [4, 5, 6, 2]:
    ax1.plot(x, np.array(shares[c]) * 100, "o-", ms=4, lw=1.7,
             color=cmap(c), label=names[c])
ax1.set_ylabel("占当月提问的比例（%）", fontsize=11)
ax1.set_title("变化最显著的四个簇", fontsize=11.5, pad=8)
ax1.grid(alpha=0.3, ls="--")
ax1.legend(fontsize=8.5, loc="upper center", ncol=2, framealpha=0.9)
ax1.set_ylim(-2, 46)
ax1.set_xticks(x)
ax1.set_xticklabels([m[2:] for m in months], rotation=45, fontsize=9)

ax2 = fig.add_subplot(gs[1, 1])
for c in [0, 1, 3]:
    ax2.plot(x, np.array(shares[c]) * 100, "o-", ms=4, lw=1.7,
             color=cmap(c), label=names[c])
ax2.set_ylabel("占当月提问的比例（%）", fontsize=11)
ax2.set_title("变化平缓的三个簇", fontsize=11.5, pad=8)
ax2.grid(alpha=0.3, ls="--")
ax2.legend(fontsize=8.5, loc="upper center", ncol=2, framealpha=0.9)
ax2.set_ylim(-2, 46)
ax2.set_xticks(x)
ax2.set_xticklabels([m[2:] for m in months], rotation=45, fontsize=9)

plt.savefig("figures/topic_over_time_k7.png",
            dpi=160, bbox_inches="tight")
print("saved")
