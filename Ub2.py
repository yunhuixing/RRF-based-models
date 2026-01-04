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

MODEL_TAG = "dynrrf_close"
ENABLE_2D_PLOT = False

CALIBRATION_KEYWORDS = ["CALA", "CALB", "CALC", "CALD"]

df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

cal_points = df[df["model point"].astype(str).str.upper().isin(CALIBRATION_KEYWORDS)].copy()
test_points = df.loc[~df.index.isin(cal_points.index)].copy()

X2_means = (
    cal_points
    .groupby("model point")["X2"]
    .mean()
    .sort_values()
)

levels = list(X2_means.index)
X2_vals = list(X2_means.values)

def fit_interval(low_idx, high_idx=None):
    if high_idx is None:
        subset = cal_points[cal_points["model point"] == levels[low_idx]]
    else:
        subset = cal_points[
            cal_points["model point"].isin([levels[low_idx], levels[high_idx]])
        ]
    model = LinearRegression()
    model.fit(subset[["X1"]].values, subset["RRF"].values)
    return model

interval_models = {
    "low": fit_interval(0),
    "CALA-CALB": fit_interval(0, 1),
    "CALB-CALC": fit_interval(1, 2),
    "CALC-CALD": fit_interval(2, 3),
    "high": fit_interval(3)
}

def predict_rrf(x1, x2):
    if x2 < X2_vals[0]:
        model = interval_models["low"]
    elif x2 <= X2_vals[1]:
        model = interval_models["CALA-CALB"]
    elif x2 <= X2_vals[2]:
        model = interval_models["CALB-CALC"]
    elif x2 <= X2_vals[3]:
        model = interval_models["CALC-CALD"]
    else:
        model = interval_models["high"]
    return model.predict(np.array(x1).reshape(-1, 1))[0]

df["Pred_RRF"] = df.apply(
    lambda r: predict_rrf(r["X1"], r["X2"]), axis=1
)

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

print("✔ DynRRF, Close back-calculation completed.")
print(f"✔ Results saved to: {OUT_DIR.resolve()}")

def plot_2d(interval_models, cal_df, test_df, output_path):
    X1_range = np.linspace(
        cal_df["X1"].min(), cal_df["X1"].max(), 200
    )

    plt.figure(figsize=(8, 6))

    for mp, g in cal_df.groupby("model point"):
        plt.scatter(g["X1"], g["RRF"], s=25)

    colors = ["red", "orange", "green", "blue", "purple"]
    for (key, model), c in zip(interval_models.items(), colors):
        plt.plot(
            X1_range,
            model.predict(X1_range.reshape(-1, 1)),
            lw=2,
            color=c
        )

    plt.scatter(
        test_df["X1"], test_df["Pred_RRF"],
        c="black", marker="x", s=30
    )

    plt.xlabel("X1")
    plt.ylabel("RRF")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if ENABLE_2D_PLOT:
    plot_2d(
        interval_models,
        cal_points,
        df[df["Set"] == "Test"],
        OUT_DIR / f"{MODEL_TAG}_fit_2d.png"
    )
