import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# RESULTS DIRECTORY
# =====================================================

RESULT_DIR = "results"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

# =====================================================
# INPUT HASIL PENELITIAN
# =====================================================

results = {

    "Model": [
        "MobileNetV2",
        "ResNet50",
        "ResNet101"
    ],

    "Accuracy": [
        94.50,
        94.67,
        93.33
    ],

    "Precision": [
        95.00,
        95.00,
        93.00
    ],

    "Recall": [
        94.00,
        95.00,
        93.00
    ],

    "F1-Score": [
        95.00,
        95.00,
        93.00
    ]
}

# =====================================================
# DATAFRAME
# =====================================================

df = pd.DataFrame(results)

# =====================================================
# SORT BERDASARKAN ACCURACY
# =====================================================

df = df.sort_values(
    by="Accuracy",
    ascending=False
)

# =====================================================
# SAVE TABLE
# =====================================================

csv_path = os.path.join(
    RESULT_DIR,
    "model_comparison.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print("\nModel Comparison Table\n")
print(df)

print(
    f"\nSaved : {csv_path}"
)

# =====================================================
# BEST MODEL
# =====================================================

best_model = df.iloc[0]

print("\n==========================")
print("BEST MODEL")
print("==========================")

print(
    f"Model    : {best_model['Model']}"
)

print(
    f"Accuracy : {best_model['Accuracy']:.2f}%"
)

# =====================================================
# ACCURACY BAR CHART
# =====================================================

plt.figure(
    figsize=(8,5)
)

plt.bar(
    df["Model"],
    df["Accuracy"]
)

plt.title(
    "Model Accuracy Comparison"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.ylim(
    80,
    100
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "accuracy_comparison.png"
    ),
    dpi=300
)

plt.close()

# =====================================================
# METRICS COMPARISON
# =====================================================

plt.figure(
    figsize=(10,6)
)

x = range(len(df))

width = 0.2

plt.bar(
    [i-width for i in x],
    df["Precision"],
    width=width,
    label="Precision"
)

plt.bar(
    x,
    df["Recall"],
    width=width,
    label="Recall"
)

plt.bar(
    [i+width for i in x],
    df["F1-Score"],
    width=width,
    label="F1-Score"
)

plt.xticks(
    x,
    df["Model"]
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "Precision Recall F1 Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "metrics_comparison.png"
    ),
    dpi=300
)

plt.close()

# =====================================================
# SUMMARY TXT
# =====================================================

summary_path = os.path.join(
    RESULT_DIR,
    "research_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PEATLAND CLASSIFICATION MODEL COMPARISON\n\n"
    )

    f.write(
        df.to_string(index=False)
    )

    f.write("\n\n")

    f.write(
        f"Best Model : {best_model['Model']}\n"
    )

    f.write(
        f"Accuracy   : {best_model['Accuracy']:.2f}%\n"
    )

print(
    f"Saved : {summary_path}"
)

print("\nDONE")