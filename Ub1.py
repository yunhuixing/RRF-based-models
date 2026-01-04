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

MODEL_TAG = "dynrrf_all"
ENABLE_2D_PLOT = False

CALIBRATION_KEYWORDS = ["CALA", "CALB", "CALC", "CALD"]

df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

train_data = df[df["model point"].astype(str).str.upper().isin(CALIBRATION_KEYWORDS)].copy()
test_data = df.loc[~df.index.isin(train_data.index)].copy()

X1_train = train_data[["X1"]].values
RRF_train = train_data["RRF"].values

lin_model = LinearRegression()
lin_model.fit(X1_train, RRF_train)

def back_calculate(df, label):
    df = df.copy()
    df["Pred_RRF"] = lin_model.predict(df[["X1"]].values)
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

print("✔ DynRRF, All back-calculation completed.")
print(f"✔ Results saved to: {OUT_DIR.resolve()}")

def plot_2d_fit(train_df, all_df, output_path):
    X1_range = np.linspace(all_df["X1"].min(), all_df["X1"].max(), 200).reshape(-1, 1)
    RRF_fit = lin_model.predict(X1_range)

    plt.figure(figsize=(8, 6))
    plt.scatter(train_df["X1"], train_df["RRF"], c="blue", s=25)
    plt.plot(X1_range, RRF_fit, c="red", lw=2)
    plt.scatter(all_df["X1"], all_df["Pred_RRF"], c="black", marker="x", s=30)
    plt.xlabel("X1")
    plt.ylabel("RRF")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if ENABLE_2D_PLOT:
    plot_2d_fit(
        train_results,
        all_results,
        OUT_DIR / f"{MODEL_TAG}_fit_2d.png"
    )
