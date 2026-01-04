import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path
import matplotlib

matplotlib.rcParams['font.family'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.sans-serif'] = ['SimSun']

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

file_path = DATA_DIR / "oxc.xlsx"
df = pd.read_excel(file_path, sheet_name="oxc")

df.columns = [c.strip() for c in df.columns]

mp = df.get("model point", pd.Series([np.nan] * len(df)))
is_cal = (
    mp.notna()
    & mp.astype(str).str.strip().ne("")
    & mp.astype(str).str.lower().ne("nan")
)

cal_points = df[is_cal].copy()

for col in ["X2", "RRF", "Actual_Concentration"]:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

X = cal_points["X2"].values.reshape(-1, 1)
y = cal_points["RRF"].values

X_quad = np.hstack([X, X**2])

model = LinearRegression()
model.fit(X_quad, y)

a = model.intercept_
b = model.coef_[0]
c = model.coef_[1]

print(f"StatCal (quadratic) fitted:")
print(f"RRF = {a:.3e} + {b:.3e}·X2 + {c:.3e}·X2²")

X_all = df["X2"].values.reshape(-1, 1)
X_all_quad = np.hstack([X_all, X_all**2])

df["Pred_RRF"] = model.predict(X_all_quad)
df["Calcback_Concentration"] = df["X2"] / df["Pred_RRF"]

den = df["Actual_Concentration"].replace(0, np.nan)
df["Accuracy_%"] = df["Calcback_Concentration"] * 100 / den

out_xlsx = OUT_DIR / "StatCal_a_plus_bX2_plus_cX2sq_results.xlsx"
df.to_excel(out_xlsx, index=False)

print(f"Results saved to: {out_xlsx}")

ENABLE_PLOT = True

if ENABLE_PLOT:
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        cal_points["X2"],
        cal_points["RRF"],
        s=28,
        alpha=0.85
    )

    x2_min, x2_max = df["X2"].min(), df["X2"].max()
    X2_line = np.linspace(x2_min, x2_max, 300).reshape(-1, 1)
    X2_line_quad = np.hstack([X2_line, X2_line**2])
    RRF_line = model.predict(X2_line_quad)

    ax.plot(X2_line, RRF_line, color="red", lw=2)

    ax.scatter(
        df["X2"],
        df["Pred_RRF"],
        color="black",
        marker="x",
        s=36
    )

    ax.set_xlabel("X2", fontsize=36)
    ax.set_ylabel("RRF", fontsize=36)
    ax.tick_params(axis="both", labelsize=32)

    plt.tight_layout()
    fig_path = OUT_DIR / "StatCal_a_plus_bX2_plus_cX2sq_2D.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {fig_path}")
