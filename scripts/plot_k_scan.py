import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 两轮扫描合并，k 为全部偶数，步长 2
odd = {4: 0.073024, 8: 0.075700, 12: 0.069558, 16: 0.063506, 20: 0.066120,
       24: 0.064741, 28: 0.062971, 32: 0.065453, 36: 0.067677, 40: 0.065561,
       44: 0.064932, 48: 0.049281, 52: 0.049793, 56: 0.051368, 60: 0.066563,
       64: 0.049471}
even = {2: 0.060288, 6: 0.077390, 10: 0.066993, 14: 0.067327, 18: 0.064527,
        22: 0.064778, 26: 0.065537, 30: 0.047157, 34: 0.046734, 38: 0.065891,
        42: 0.064900, 46: 0.047851, 50: 0.064601, 54: 0.065443, 58: 0.064506,
        62: 0.065326}
merged = {**odd, **even}
ks = np.array(sorted(merged))
vals = np.array([merged[k] for k in ks])

w = {4: 24599, 8: 22550, 12: 21604, 16: 20925, 20: 20366, 24: 19960,
     28: 19653, 32: 19363, 36: 19089, 40: 18864, 44: 18686, 48: 18515,
     52: 18339, 56: 18181, 60: 18037, 64: 17924,
     2: 27352, 6: 23228, 10: 22041, 14: 21254, 18: 20778, 22: 20205,
     26: 19788, 30: 19514, 34: 19226, 38: 18955, 42: 18777, 46: 18597,
     50: 18423, 54: 18280, 58: 18147, 62: 17972}
wssse = np.array([w[k] for k in ks])

smooth = np.array([vals[max(0, i - 1):min(len(vals), i + 2)].mean()
                   for i in range(len(vals))])

fig, axes = plt.subplots(2, 1, figsize=(12, 9.5), sharex=True,
                         gridspec_kw={"height_ratios": [3, 2]})

ax = axes[0]
ax.plot(ks, vals, "o-", color="#1f77b4", ms=5, lw=1.3, label="轮廓系数")
ax.plot(ks, smooth, "--", color="#d62728", lw=2.0, alpha=0.85,
        label="滑动平均（窗口 3）")
ax.axvspan(2, 16, color="#2ca02c", alpha=0.12, label="高位区 k = 2–16")
ax.axhspan(vals.min(), 0.0505, color="#ff7f0e", alpha=0.10,
           label="低位带 < 0.0505")

peak_i = int(np.argmax(vals))
ax.annotate(f"峰值 k=6\n{vals[peak_i]:.4f}",
            xy=(ks[peak_i], vals[peak_i]),
            xytext=(ks[peak_i] + 3.0, vals[peak_i] + 0.0028),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2))

sm_i = int(np.argmax(smooth))
ax.annotate(f"平滑峰值 k={ks[sm_i]}",
            xy=(ks[sm_i], smooth[sm_i]),
            xytext=(ks[sm_i] + 3.5, smooth[sm_i] + 0.0065),
            fontsize=9, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))

ax.set_ylabel("轮廓系数", fontsize=12)
ax.set_title("10 万条抽样：轮廓系数随 k 的变化（k = 2–64，步长 2，10,000 点固定采样）",
             fontsize=13, pad=12)
ax.grid(alpha=0.3, ls="--")
ax.legend(fontsize=9.5, loc="lower left", framealpha=0.92)
ax.set_ylim(0.044, 0.081)

ax2 = axes[1]
ax2.plot(ks, wssse / 1000, "s-", color="#8c564b", ms=4.5, lw=1.3,
         label="WSSSE（组内平方和）")
ax2.set_ylabel("WSSSE（千）", fontsize=12)
ax2.set_xlabel("k", fontsize=12)
ax2.grid(alpha=0.3, ls="--")
ax2.legend(fontsize=9.5, loc="upper right", framealpha=0.92)
ax2.set_xticks(range(2, 65, 2))

for a in axes:
    a.tick_params(labelsize=9.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG, "k_scan_step2.png"),
            dpi=160, bbox_inches="tight")
print("saved")
