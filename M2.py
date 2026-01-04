import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from matplotlib import colors

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "oxc.xlsx"
SHEET_NAME = "oxc"

MODEL_TAG = "lowess"
ENABLE_3D_PLOT = False
ENABLE_2D_PLOT = False

CALIBRATION_KEYWORDS = ["CALA", "CALB", "CALC", "CALD"]

LOWESS_FRAC = 0.3
LOWESS_IT = 1

df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

train_data = df[df["model point"].astype(str).str.upper().isin(CALIBRATION_KEYWORDS)].copy()
test_data = df.loc[~df.index.isin(train_data.index)].copy()

x_train = train_data["X2"].values
y_train = train_data["RRF"].values

lowess_fit = lowess(
    endog=y_train,
    exog=x_train,
    frac=LOWESS_FRAC,
    it=LOWESS_IT,
    return_sorted=True
)

x_fit, y_fit = lowess_fit[:, 0], lowess_fit[:, 1]

def lowess_predict(X):
    if isinstance(X, pd.DataFrame):
        x = X["X2"].values
    else:
        x = np.asarray(X)
    return np.interp(x, x_fit, y_fit)

def back_calculate(df, label):
    df = df.copy()
    df["Pred_RRF"] = lowess_predict(df)
    df["Back_Calculated_C"] = df["X2"] / df["Pred_RRF"]
    df["Relative_Error_%"] = 100 * (
        df["Back_Calculated_C"] - df["Actual_Concentration"]
    ) / df["Actual_Concentration"]
    df["Accuracy_%"] = 100 - df["Relative_Error_%"].abs()
    df["Set"] = label
    return df

train_results = back_calculate(train_data, "Calibration")
test_results = back_calculate(test_data, "Test")

all_results = pd.concat([train_results, test_results], ignore_index=True)

train_accuracy = (
    train_results
    .groupby("model point")
    .apply(lambda g: pd.Series({
        "Mean_RE_%": g["Relative_Error_%"].mean(),
        "SD_RE_%": g["Relative_Error_%"].std(),
        "Mean_Accuracy_%": g["Accuracy_%"].mean(),
        "n": len(g)
    }))
    .reset_index()
)

all_results.to_excel(
    OUT_DIR / f"{MODEL_TAG}_all_results.xlsx",
    index=False
)

train_accuracy.to_excel(
    OUT_DIR / f"{MODEL_TAG}_model_point_accuracy.xlsx",
    index=False
)

print(f"LOWESS (frac={LOWESS_FRAC}, it={LOWESS_IT}) back-calculation completed.")
print(f"Results saved to: {OUT_DIR.resolve()}")

def plot_3d_surface(results_df, output_path):
    time_range = np.linspace(results_df["X1"].min(), results_df["X1"].max(), 50)
    rrfc_range = np.linspace(results_df["X2"].min(), results_df["X2"].max(), 50)
    T_grid, RRFC_grid = np.meshgrid(time_range, rrfc_range)

    Pred_grid = np.interp(RRFC_grid, x_fit, y_fit)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = colors.LinearSegmentedColormap.from_list(
        "blue_yellow", ["blue", "yellow"]
    )

    ax.plot_surface(
        T_grid, RRFC_grid, Pred_grid,
        cmap=cmap, alpha=0.9, linewidth=0
    )

    ax.scatter(
        results_df["X1"], results_df["X2"], results_df["RRF"],
        c="black", s=6
    )

    ax.scatter(
        results_df["X1"], results_df["X2"], results_df["Pred_RRF"],
        c="orange", marker="^", s=16
    )

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("RRF")
    ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

if ENABLE_3D_PLOT:
    plot_3d_surface(
        all_results,
        OUT_DIR / f"{MODEL_TAG}_3d_surface.png"
    )

def plot_2d_slices(results_df, output_path):
    x2_levels = np.linspace(results_df["X2"].min(), results_df["X2"].max(), 3)
    x1_range = np.linspace(results_df["X1"].min(), results_df["X1"].max(), 100)

    plt.figure(figsize=(8, 6))
    for val in x2_levels:
        y = np.interp(val, x_fit, y_fit)
        plt.plot(x1_range, np.full_like(x1_range, y))

    plt.scatter(
        results_df["X1"], results_df["RRF"],
        c="gray", alpha=0.4, s=10
    )

    plt.xlabel("X1")
    plt.ylabel("RRF")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if ENABLE_2D_PLOT:
    plot_2d_slices(
        all_results,
        OUT_DIR / f"{MODEL_TAG}_2d_slices.png"
    )
