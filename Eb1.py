# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib import colors

matplotlib.rcParams["font.family"] = ["Times New Roman"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.sans-serif"] = ["SimSun"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, "oxc.xlsx")
OUT_FIG = os.path.join(OUT_DIR, "oxc_3D_enhanced.png")

df = pd.read_excel(INPUT_FILE, sheet_name="oxc")
df.columns = [c.strip() for c in df.columns]

train_df = df[df["model point"].astype(str).str.upper().isin(
    ["CALA", "CALB", "CALC", "CALD"]
)].copy()
test_df = df.drop(train_df.index).copy()

for _df in (train_df, test_df):
    _df["X1_sq"] = _df["X1"] ** 2
    _df["X2_sq"] = _df["X2"] ** 2

X = sm.add_constant(train_df[["X1", "X2", "X1_sq", "X2_sq"]])
y = train_df["RRF"]
model = sm.OLS(y, X).fit()

for _df in (train_df, test_df):
    Xp = sm.add_constant(_df[["X1", "X2", "X1_sq", "X2_sq"]])
    _df["Pred_RRF"] = model.predict(Xp)

all_results = pd.concat([train_df, test_df], ignore_index=True)

x1_range = np.linspace(df["X1"].min(), df["X1"].max(), 200)
x2_range = np.linspace(df["X2"].min(), df["X2"].max(), 200)
X1g, X2g = np.meshgrid(x1_range, x2_range)

Xg = pd.DataFrame({
    "X1": X1g.ravel(),
    "X2": X2g.ravel(),
    "X1_sq": X1g.ravel() ** 2,
    "X2_sq": X2g.ravel() ** 2
})
Xg = sm.add_constant(Xg)
Zg = model.predict(Xg).values.reshape(X1g.shape)

cmap = colors.LinearSegmentedColormap.from_list(
    "blue_yellow", ["blue", "yellow"]
)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X1g, X2g, Zg,
    cmap=cmap, alpha=0.9, linewidth=0
)

ax.scatter(
    all_results["X1"], all_results["X2"], all_results["RRF"],
    c="black", s=6, depthshade=False
)

ax.scatter(
    all_results["X1"], all_results["X2"], all_results["Pred_RRF"],
    c="orange", marker="^", s=18
)

for _, r in all_results.iterrows():
    ax.plot(
        [r["X1"], r["X1"]],
        [r["X2"], r["X2"]],
        [r["RRF"], r["Pred_RRF"]],
        color="gray", alpha=0.6, linewidth=1
    )

ax.set_xlabel("X1", fontsize=36, labelpad=30)
ax.set_ylabel("X2", fontsize=36, labelpad=30)
ax.tick_params(axis="both", labelsize=32)
ax.view_init(elev=30, azim=135)

cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label("RRF", fontsize=36)
cbar.ax.tick_params(labelsize=34)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.close()
