import time
import matplotlib.pyplot as plt
import pandas as pd
from DataProcessing_module import (
    load_dataset,
    get_balanced_subset,
    extract_features_and_labels,
    split_data
)
from RandomForest_module import train_random_forest, evaluate_model

# --- Config
dataset_path = "Noisy_Iraq_dataset.csv"
labels = [0, 1, 2]
model_feature = [1, 2, 3, 4, 5]
combined_all = []

# --- Preprocessing
full_data = load_dataset(dataset_path)
subset_data = get_balanced_subset(full_data)

X1, Y1 = extract_features_and_labels(full_data)
X2, Y2 = extract_features_and_labels(subset_data)

X1_train, X1_test, Y1_train, Y1_test = split_data(X1, Y1)
X2_train, X2_test, Y2_train, Y2_test = split_data(X2, Y2)

# --- Training and Evaluation Loop
for feature in model_feature:
    start_time = time.time()

    clf1 = train_random_forest(X1_train, Y1_train, max_features=feature)
    clf2 = train_random_forest(X2_train, Y2_train, max_features=feature)

    Y1_pred = clf1.predict(X1_test)
    Y2_pred = clf2.predict(X2_test)

    df1, acc1 = evaluate_model(Y1_test, Y1_pred, labels)
    df2, acc2 = evaluate_model(Y2_test, Y2_pred, labels)

    print(f"Model with {feature} features (Full data): {acc1 * 100:.2f}%")
    print(f"Model with {feature} features (53 samples): {acc2 * 100:.2f}%")
    print(f"Time: {time.time() - start_time:.2f} seconds\n")

    combined = pd.concat([df1, df2], ignore_index=True)
    combined_all.append(combined)

# --- Aggregate Results
final_df = pd.concat(combined_all, ignore_index=True)
models = ["All samples", "53 samples"]
metrics = ["specificity", "recall", "precision", "f_score"]

final_df["model"] = [models[i // 3 % 2] for i in range(len(final_df))]
final_df["trees"] = [tree for tree in model_feature for _ in range(6)]

# --- Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    for cls in labels:
        for model in models:
            subset = final_df[(final_df["label"] == cls) & (final_df["model"] == model)]
            ax.plot(subset["trees"], subset[metric], marker='o',
                    label=f"{model} - Class {cls}",
                    linewidth=2.5, alpha=0.8)
    ax.set_title(f"{metric.capitalize()}")
    ax.set_xlabel("ilosc cech")
    ax.set_ylabel("wartosc")
    ax.set_xticks(model_feature)
    ax.grid(True, axis='x')
    ax.legend()

plt.suptitle("Model Performance vs. Number of Features", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()