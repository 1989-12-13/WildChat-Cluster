import theme  # noqa: F401  统一图表配色
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

langs = ["English", "Chinese", "Russian", "French", "Spanish", "Arabic",
         "German", "Portuguese", "Latin", "Italian", "Turkish", "Vietnamese",
         "Korean", "Persian", "Japanese", "Polish", "Indonesian", "Dutch"]
totals = [980259, 329474, 217812, 66137, 52692, 30370, 29369, 24964, 15829,
          15197, 14827, 12476, 12458, 11015, 10584, 10562, 8259, 6528]

# 每语言内七个簇的占比，顺序 [0,1,2,3,4,5,6]
raw = {
    "English":   [0.2836, 0.1955, 0.1861, 0.0030, 0.0096, 0.0840, 0.2382],
    "Chinese":   [0.0313, 0.0163, 0.1920, 0.0126, 0.6492, 0.0005, 0.0980],
    "Russian":   [0.2216, 0.0617, 0.2424, 0.0126, 0.0030, 0.0002, 0.4585],
    "French":    [0.5043, 0.0361, 0.1420, 0.0106, 0.0087, 0.0000, 0.2983],
    "Spanish":   [0.3555, 0.0815, 0.1378, 0.0041, 0.0209, 0.0002, 0.4000],
    "Arabic":    [0.4171, 0.0787, 0.0562, 0.0294, 0.0492, 0.0001, 0.3694],
    "German":    [0.4857, 0.0894, 0.1455, 0.0043, 0.0203, 0.0001, 0.2547],
    "Portuguese":[0.3176, 0.1019, 0.1090, 0.0028, 0.0116, 0.0002, 0.4570],
    "Latin":     [0.4726, 0.0530, 0.2823, 0.0021, 0.0091, 0.0000, 0.1809],
    "Italian":   [0.3453, 0.1026, 0.1876, 0.0228, 0.0085, 0.0001, 0.3332],
    "Turkish":   [0.3400, 0.0848, 0.1161, 0.0035, 0.0023, 0.0000, 0.4533],
    "Vietnamese":[0.2305, 0.0249, 0.2086, 0.0186, 0.0650, 0.0000, 0.4524],
    "Korean":    [0.3061, 0.1753, 0.1154, 0.0104, 0.0130, 0.0002, 0.3797],
    "Persian":   [0.3935, 0.0340, 0.0723, 0.0607, 0.0011, 0.0001, 0.4384],
    "Japanese":  [0.4498, 0.2053, 0.0291, 0.0079, 0.0735, 0.0008, 0.2336],
    "Polish":    [0.4320, 0.0724, 0.1564, 0.0135, 0.0034, 0.0000, 0.3222],
    "Indonesian":[0.2357, 0.0331, 0.1200, 0.0016, 0.0084, 0.0000, 0.6013],
    "Dutch":     [0.3695, 0.0492, 0.2085, 0.0043, 0.0236, 0.0000, 0.3450],
}
cnames = ["簇0 超短指令碎片", "簇1 同人剧本", "簇2 代码", "簇3 问候",
          "簇4 中文短问", "簇5 Midjourney", "簇6 学术作业"]

mat = np.array([raw[l] for l in langs])
labels = [f"{l} ({t//1000}k)" for l, t in zip(langs, totals)]

fig, axes = plt.subplots(1, 2, figsize=(15, 8.2),
                         gridspec_kw={"width_ratios": [1.45, 1]})

ax = axes[0]
im = ax.imshow(mat * 100, aspect="auto", cmap="YlOrRd", vmin=0, vmax=65)
ax.set_xticks(range(7))
ax.set_xticklabels(cnames, rotation=38, ha="right", fontsize=9.5)
ax.set_yticks(range(len(langs)))
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_title("各语言内部、七个簇的占比（%）", fontsize=12.5, pad=10)

for i in range(len(langs)):
    for j in range(7):
        v = mat[i, j] * 100
        if v >= 1.0:
            color = "white" if v > 35 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=8.2, color=color)

plt.colorbar(im, ax=ax, fraction=0.032, pad=0.02, label="占比（%）")

ax2 = axes[1]
ent = [3.030, 1.447, 2.518, 3.438, 0.610, 0.034, 2.833]
bar_colors = [theme.PRIMARY] * 7
ax2.barh(range(7), ent, color=bar_colors, height=0.62)
ax2.set_yticks(range(7))
ax2.set_yticklabels(cnames, fontsize=9.5)
ax2.invert_yaxis()
ax2.set_xlabel("语言熵（bits）", fontsize=11)
ax2.set_title("各簇的语言分散程度", fontsize=12.5, pad=10)
ax2.grid(alpha=0.3, ls="--", axis="x")
ax2.set_xlim(0, 4.0)
for i, v in enumerate(ent):
    ax2.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9.5)

ax2.annotate("簇5 几乎纯英文\n（仅 15 种语言）",
             xy=(0.034, 5), xytext=(1.15, 4.55), fontsize=9,
             arrowprops=dict(arrowstyle="->", lw=1.1))
ax2.annotate("簇4 几乎纯中文\n（熵最低的非单语簇）",
             xy=(0.610, 4), xytext=(1.5, 2.55), fontsize=9,
             arrowprops=dict(arrowstyle="->", lw=1.1))

plt.tight_layout()
plt.savefig(os.path.join(FIG, "lang_topic_k7.png"),
            dpi=155, bbox_inches="tight")
print("saved")
