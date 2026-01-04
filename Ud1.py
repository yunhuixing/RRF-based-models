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

def assign_batch(x1):
    if 1 <= x1 <= 40:
        return 1
    elif 41 <= x1 <= 80:
        return 2
    elif 81 <= x1 <= 120:
        return 3
    elif 121 <= x1 <= 160:
        return 4
    elif 161 <= x1 <= 200:
        return 5
    elif 201 <= x1 <= 240:
        return 6
    else:
        return np.nan

df["Batch"] = df["X1"].apply(assign_batch)

mp = df.get("model point", pd.Series([np.nan] * len(df)))
is_cal = (
    mp.notna()
    & mp.astype(str).str.strip().ne("")
    & mp.astype(str).str.lower().ne("nan")
)

cal_points = df[is_cal].copy()
test_points = df[~is_cal].copy()

all_batches = sorted(cal_points["Batch"].dropna().unique())
min_batch, max_batch = min(all_batches), max(all_batches)

df["Pred_RRF"] = np.nan

for idx, row in test_points.iterrows():
    batch_test = int(row["Batch"])

    if batch_test == min_batch:
        batches = [min_batch, min_batch + 1]
    elif batch_test == max_batch:
        batches = [max_batch - 1, max_batch]
    else:
        batches = [batch_test - 1, batch_test + 1]

    subset = cal_points[cal_points["Batch"].isin(batches)]
    if subset.empty:
        continue

    X = subset["X2"].values.reshape(-1, 1)
    y = subset["RRF"].values

    model = LinearRegression()
    model.fit(X, y)

    mask = df["Batch"].isin(batches)
    X_all = df.loc[mask, "X2"].values.reshape(-1, 1)
    df.loc[mask, "Pred_RRF"] = model.predict(X_all)

df["Calcback_Concentration"] = df["X2"] / df["Pred_RRF"]

den = df["Actual_Concentration"].replace(0, np.nan)
df["Accuracy_%"] = df["Calcback_Concentration"] * 100 / den

out_xlsx = OUT_DIR / "DynCal_a_plus_bX2_results.xlsx"
df.to_excel(out_xlsx, index=False)

print(f"Results saved to: {out_xlsx}")

ENABLE_PLOT = True

if ENABLE_PLOT:
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        cal_points["X2"],
        cal_points["RRF"],
        s=30,
        alpha=0.8
    )

    ax.scatter(
        df.loc[~is_cal, "X2"],
        df.loc[~is_cal, "Pred_RRF"],
        color="black",
        marker="x",
        s=40
    )

    ax.set_xlabel("X2", fontsize=36)
    ax.set_ylabel("RRF", fontsize=36)
    ax.tick_params(axis="both", labelsize=32)

    plt.tight_layout()
    fig_path = OUT_DIR / "DynCal_a_plus_bX2_2D.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {fig_path}")
