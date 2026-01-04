import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "oxc.xlsx"
SHEET_NAME = "oxc"

MODEL_TAG = "statrrf_all"

CALIBRATION_KEYWORDS = ["CALA", "CALB", "CALC", "CALD"]

df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

train_data = df[df["model point"].astype(str).str.upper().isin(CALIBRATION_KEYWORDS)].copy()
test_data = df.loc[~df.index.isin(train_data.index)].copy()

mean_rrf = train_data["RRF"].mean()
print(f"✔ Mean RRF from calibration samples: {mean_rrf:.6g}")

def back_calculate(df, label):
    df = df.copy()
    df["Pred_RRF"] = mean_rrf
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


