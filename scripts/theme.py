import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.facecolor"] = "#FAF7F2"
matplotlib.rcParams["savefig.facecolor"] = "#FAF7F2"
matplotlib.rcParams["axes.facecolor"] = "#FAF7F2"
matplotlib.rcParams["text.color"] = "#44423F"
matplotlib.rcParams["axes.labelcolor"] = "#44423F"
matplotlib.rcParams["axes.edgecolor"] = "#B8AFA2"
matplotlib.rcParams["xtick.color"] = "#5A554D"
matplotlib.rcParams["ytick.color"] = "#5A554D"
matplotlib.rcParams["axes.titlecolor"] = "#1C1B1A"
matplotlib.rcParams["grid.color"] = "#C9BFB2"
matplotlib.rcParams["legend.facecolor"] = "#FAF7F2"
matplotlib.rcParams["legend.edgecolor"] = "#C9BFB2"

# 七个簇的系列色：暖色为主，饱和度统一，彼此可区分
PALETTE = [
    "#8C3A2C",  # 深砖红
    "#C8932A",  # 芥末黄
    "#2F5D50",  # 墨绿
    "#B0563C",  # 暖赭
    "#6B8E8A",  # 灰青
    "#7A5C86",  # 暗紫
    "#4A4A48",  # 深灰
]

# 主要数据线
PRIMARY = "#8C3A2C"
SECONDARY = "#2F5D50"
ACCENT = "#C8932A"

print("配色已统一")

