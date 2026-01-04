import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "oxc.xlsx"
SHEET_NAME = "oxc"

MODEL_TAG = "dynrrf_closest"
ENABLE_2D_PLOT = False

CALIBRATION_KEYWORDS = ["CALA", "CALB", "CALC", "CALD"]

df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

cal_points = df[df["model point"].astype(str).str.upper().isin(CALIBRATION_KEYWORDS)].copy()
test_points = df.loc[~df.index.isin(cal_points.index)].copy()

X2_means = cal_points.groupby("model point")["X2"].mean()

def get_nearest_level(x2_value):
    return (X2_means - x2_value).abs().idxmin()

level_models = {}
for level, group in cal_points.groupby("model point"):
    model = LinearRegression()
    model.fit(group[["X1"]].values, group["RRF"].values)
    level_models[level] = model

pred_rrf_list = []
nearest_level_list = []

for _, row in df.iterrows():
    x1, x2 = row["X1"], row["X2"]

    if pd.notna(row["model point"]):
        level = row["model point"]
    else:
        level = get_nearest_level(x2)

    model = level_models[level]
    pred_rrf = model.predict(np.array(x1).reshape(-1, 1))[0]

    pred_rrf_list.append(pred_rrf)
    nearest_level_list.append(level)

df["Pred_RRF"] = pred_rrf_list
df["Nearest_Level"] = nearest_level_list

df["Back_Calculated_C"] = df["X2"] / df["Pred_RRF"]
df["Relative_Error_%"] = 100 * (
    df["Back_Calculated_C"] - df["Actual_Concentration"]
) / df["Actual_Concentration"]
df["Accuracy_%"] = 100 - df["Relative_Error_%"].abs()

df["Set"] = np.where(
    df.index.isin(cal_points.index),
    "Calibration",
    "Test"
)

out_all = OUT_DIR / f"{MODEL_TAG}_all_results.xlsx"
df.to_excel(out_all, index=False)

print("✔ DynRRF, Closest (Ub3) back-calculation completed.")
print(f"✔ Results saved to: {OUT_DIR.resolve()}")

def plot_2d(level_models, cal_df, full_df, output_path):
    X1_range = np.linspace(
        cal_df["X1"].min(), cal_df["X1"].max(), 200
    ).reshape(-1, 1)

    plt.figure(figsize=(8, 6))

    for level, g in cal_df.groupby("model point"):
        plt.scatter(g["X1"], g["RRF"], s=25)

    for level, model in level_models.items():
        plt.plot(
            X1_range,
            model.predict(X1_range),
            lw=2
        )

    plt.scatter(
        full_df["X1"],
        full_df["Pred_RRF"],
        c="black",
        marker="x",
        s=30
    )

    plt.xlabel("X1")
    plt.ylabel("RRF")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if ENABLE_2D_PLOT:
    plot_2d(
        level_models,
        cal_points,
        df,
        OUT_DIR / f"{MODEL_TAG}_fit_2d.png"
    )
