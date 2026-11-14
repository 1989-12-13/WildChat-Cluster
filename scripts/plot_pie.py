import theme  # noqa: F401  统一图表配色
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

sizes = {0: 509601, 6: 494026, 2: 355876, 1: 237856,
         4: 231478, 5: 82582, 3: 31608}
pct = {0: "26.2", 6: "25.4", 2: "18.3", 1: "12.2",
       4: "11.9", 5: "4.2", 3: "1.6"}
order = [0, 6, 2, 1, 4, 5, 3]

fig, ax = plt.subplots(figsize=(6.2, 4.9))
wedges, _ = ax.pie(
    [sizes[c] for c in order],
    colors=[theme.PALETTE[c] for c in order],
    startangle=90,
    counterclock=False,
    radius=1.0,
    wedgeprops=dict(width=0.44, edgecolor="#FAF7F2", linewidth=2.4),
)

for c, w in zip(order, wedges):
    if float(pct[c]) < 10:
        continue
    ang = (w.theta1 + w.theta2) / 2
    ax.text(0.78 * np.cos(np.deg2rad(ang)), 0.78 * np.sin(np.deg2rad(ang)),
            f"{pct[c]}%", ha="center", va="center",
            fontsize=15, weight="bold", color="white")

SMALL = {5: (-1.02, 1.06), 3: (0.02, 1.24)}
for c, (tx, ty) in SMALL.items():
    w = wedges[order.index(c)]
    ang = (w.theta1 + w.theta2) / 2
    x0, y0 = 1.02 * np.cos(np.deg2rad(ang)), 1.02 * np.sin(np.deg2rad(ang))
    ax.annotate(f"{pct[c]}%", xy=(x0, y0), xytext=(tx, ty),
                fontsize=12, weight="bold", color=theme.PALETTE[c],
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=theme.PALETTE[c],
                                lw=1.1, shrinkA=2, shrinkB=4))

ax.text(0, 0.10, "1,943,027", ha="center", va="center",
        fontsize=17, weight="bold", color="#1C1B1A")
ax.text(0, -0.16, "条用户提问", ha="center", va="center",
        fontsize=10, color="#6B655C")

ax.set_aspect("equal")
ax.set_xlim(-1.32, 1.28)
ax.set_ylim(-1.12, 1.34)
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig("figures/cluster_share_pie.png",
            dpi=200, facecolor="#FAF7F2", bbox_inches="tight", pad_inches=0.06)
print("saved")
