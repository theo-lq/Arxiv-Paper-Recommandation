import numpy as np
import pandas as pd


def get_file(path, learning_rate, weight_decay):
    df = pd.read_csv(path)
    df["Time"] = df["Time"].apply(lambda string: string.replace(" minutes", "")).astype(float)
    df["Weight decay"] = weight_decay
    df["Learning rate"] = learning_rate
    df = df[["Technical name", "With custom trainer", "Learning rate", "Weight decay", "Threshold", "Accuracy", "F1", "Precision", "Recall", "Time"]]
    return df



files = [
    {"path": "Evaluation_LR1E5_WD001.csv", "learning_rate": 1e-5, "weight_decay": 0.01},
    {"path": "Evaluation_LR1E5_WD005.csv", "learning_rate": 1e-5, "weight_decay": 0.05},
    {"path": "Evaluation_LR1E6_WD001.csv", "learning_rate": 1e-6, "weight_decay": 0.01},
    {"path": "Evaluation_LR1E6_WD005.csv", "learning_rate": 1e-6, "weight_decay": 0.05},
    {"path": "Evaluation_LR5E6_WD001.csv", "learning_rate": 5e-6, "weight_decay": 0.01},
    {"path": "Evaluation_LR5E6_WD005.csv", "learning_rate": 5e-6, "weight_decay": 0.05},
    {"path": "Evaluation_LR5E5_WD001.csv", "learning_rate": 5e-5, "weight_decay": 0.01},
    {"path": "Evaluation_LR5E5_WD005.csv", "learning_rate": 5e-5, "weight_decay": 0.05}
]

datasets = []
for file in files:
    df = get_file(**file)
    datasets.append(df)

df = pd.concat(datasets)
df = df.sort_values(by=["Technical name", "With custom trainer", "Learning rate", "Weight decay"])
df.to_csv("Comparison.csv", index=False)