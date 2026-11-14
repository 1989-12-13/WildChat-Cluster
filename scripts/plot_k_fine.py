import theme  # noqa: F401  统一图表配色
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 步长 1 的高位区扫描，k = 2..16
# 其中偶数与步长 2 的扫描重合，但为独立运行，数值有微小差异
ks = np.arange(2, 17)
sil = np.array([0.060759, 0.075405, 0.074679, 0.073451, 0.076877,
                0.078061, 0.066400, 0.071687, 0.065733, 0.067201,
                0.065197, 0.065010, 0.043986, 0.045580, 0.064632])
wssse = np.array([27352, 25631, 24599, 23786, 23228, 22937, 22550,
                  22230, 21996, 21771, 21555, 21372, 21403, 21206, 20963])

fig, axes = plt.subplots(2, 1, figsize=(11.5, 9), sharex=True,
                         gridspec_kw={"height_ratios": [3, 2]})

ax = axes[0]
colors = [theme.PRIMARY] * len(ks)
colors[5] = theme.ACCENT   # k=7 峰值

ax.plot(ks, sil, "-", color=theme.PRIMARY, lw=1.4, zorder=2)
ax.scatter(ks, sil, c=colors, s=80, zorder=3, edgecolors="white", linewidths=1.2)

ax.axvspan(3, 7, color=theme.SECONDARY, alpha=0.10,
           label="高位平台 k = 3–7")
ax.axvspan(13.5, 15.5, color=theme.ACCENT, alpha=0.13,
           label="低谷 k = 14–15")

iv = 5
ax.annotate(f"峰值 k=7\n{sil[iv]:.4f}",
            xy=(7, sil[iv]), xytext=(7.9, 0.0812),
            fontsize=10.5, color=theme.ACCENT,
            arrowprops=dict(arrowstyle="->", color=theme.ACCENT, lw=1.3))

ax.annotate("k=8 骤降\n-0.0117",
            xy=(8, sil[6]), xytext=(9.4, sil[6] - 0.0038),
            fontsize=9.5, color="#6B655C",
            arrowprops=dict(arrowstyle="->", color="#6B655C", lw=1.0))

ax.annotate("k=14 谷底\n0.0440",
            xy=(14, sil[12]), xytext=(11.2, sil[12] + 0.0035),
            fontsize=9.5, color=theme.SECONDARY,
            arrowprops=dict(arrowstyle="->", color=theme.SECONDARY, lw=1.0))

ax.set_ylabel("轮廓系数", fontsize=12)
ax.set_title("10 万条抽样：高位区逐点扫描（k = 2–16，步长 1，10,000 点固定采样）",
             fontsize=13, pad=12)
ax.grid(alpha=0.3, ls="--")
ax.legend(fontsize=10, loc="lower right", framealpha=0.92)
ax.set_ylim(0.040, 0.090)

ax2 = axes[1]
ax2.plot(ks, wssse / 1000, "s-", color="#6B655C", ms=5.5, lw=1.4,
         label="WSSSE（组内平方和）")
ax2.set_ylabel("WSSSE（千）", fontsize=12)
ax2.set_xlabel("k", fontsize=12)
ax2.grid(alpha=0.3, ls="--")
ax2.legend(fontsize=10, loc="upper right", framealpha=0.92)
ax2.set_xticks(ks)
ax2.annotate("k=14 处 WSSSE 回升\n（KMeans 局部最优）",
             xy=(14, wssse[12] / 1000), xytext=(9.5, 23.6),
             fontsize=9.5, color="#6B655C",
             arrowprops=dict(arrowstyle="->", color="#6B655C", lw=1.0))

for a in axes:
    a.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig("figures/k_scan_fine.png",
            dpi=160, bbox_inches="tight")
print("saved")
