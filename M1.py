import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from matplotlib import colors

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "oxc.xlsx"
SHEET_NAME = "oxc"

MODEL_TAG = "knn"
ENABLE_3D_PLOT = False

CALIBRATION_KEYWORDS = ["CALA", "CALB", "CALC", "CALD"]

df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

train_data = df[df["model point"].astype(str).str.upper().isin(CALIBRATION_KEYWORDS)].copy()
test_data = df.loc[~df.index.isin(train_data.index)].copy()

X_train = train_data[["X1", "X2"]]
y_train = train_data["RRF"]

knn_model = KNeighborsRegressor(
    n_neighbors=4,
    weights="distance"
)
knn_model.fit(X_train, y_train)

def back_calculate(df, model, label):
    df = df.copy()
    df["Pred_RRF"] = model.predict(df[["X1", "X2"]])
    df["Back_Calculated_C"] = df["X2"] / df["Pred_RRF"]
    df["Relative_Error_%"] = 100 * (
        df["Back_Calculated_C"] - df["Actual_Concentration"]
    ) / df["Actual_Concentration"]
    df["Accuracy_%"] = 100 - df["Relative_Error_%"].abs()
    df["Set"] = label
    return df

train_results = back_calculate(train_data, knn_model, "Calibration")
test_results = back_calculate(test_data, knn_model, "Test")

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

print("KNN-based RRF prediction and concentration back-calculation completed.")
print(f"Results saved to: {OUT_DIR.resolve()}")

def plot_3d_true_vs_pred(results_df, output_path):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        results_df["X1"], results_df["X2"], results_df["RRF"],
        c="black", s=8, label="Observed"
    )

    ax.scatter(
        results_df["X1"], results_df["X2"], results_df["Pred_RRF"],
        c="orange", marker="^", s=18, label="Predicted"
    )

    for _, r in results_df.iterrows():
        ax.plot(
            [r["X1"], r["X1"]],
            [r["X2"], r["X2"]],
            [r["RRF"], r["Pred_RRF"]],
            color="gray", alpha=0.5
        )

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("RRF")
    ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

if ENABLE_3D_PLOT:
    plot_3d_true_vs_pred(
        all_results,
        OUT_DIR / f"{MODEL_TAG}_3d_true_vs_pred.png"
    )
