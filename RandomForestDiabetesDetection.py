import time
from timeit import Timer
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from DataProcessing_module import (
    load_dataset,
    get_balanced_subset,
    extract_features_and_labels,
    split_data
)
from RandomForest_module import (
    train_random_forest,
    evaluate_model,
    plot_param_sweep,
    evaluate_param_sweep,
    plot_confusion_matrices
)

# --- Config
dataset_path = "Noisy_Iraq_dataset.csv"
#dataset_path = "Ready_Iraq_dataset.csv"
#dataset_path = "dataset_pca_cutoff.csv"

labels = [0, 1, 2]
param_grid = [
    ("max_features", [None, "sqrt", "log2", 0.5]),
    ("n_estimators", [10, 20, 40, 60, 80, 100]),
    ("max_depth",    [5, 10, 15, None])
]

# --- Preprocessing
full_data = load_dataset(dataset_path)
subset_data = get_balanced_subset(full_data)

X1, Y1 = extract_features_and_labels(full_data)
X2, Y2 = extract_features_and_labels(subset_data)

#X1_train, X1_test, Y1_train, Y1_test = split_data(X1, Y1)
X2_train, X2_test, Y2_train, Y2_test = split_data(X2, Y2)
start = time.time()

# --- Train model with chosen parameters
clf = train_random_forest(
    X2_train, Y2_train,
    n_estimators=40,
    max_depth=10,
    max_features= 0.6
)
end = time.time()
# --- Evaluate model
Y_pred = clf.predict(X2_test)
df_metrics, accuracy = evaluate_model(Y2_test, Y_pred, labels)

print("Evaluation metrics:")
print(df_metrics)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"time={end-start:.2f}s")

# --- Plot confusion matrix
conf_matrix = sklearn.metrics.confusion_matrix(Y2_test, Y_pred)
display_matrix = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
display_matrix.plot()
plt.title("Confusion Matrix - Default Model")
plt.tight_layout()
plt.show()




## --- Baseline confusion matrices (default params)
#clf1 = train_random_forest(X1_train, Y1_train)
#clf2 = train_random_forest(X2_train, Y2_train)
#Y1_pred = clf1.predict(X1_test)
#Y2_pred = clf2.predict(X2_test)
#plot_confusion_matrices(
#    Y1_test, Y1_pred,
#    Y2_test, Y2_pred,
#    titles=(
#        'Full Data (default params)',
#        'Subset Data (default params)'
#    )
#)
#
#for name, values in param_grid:
#
#    df = evaluate_param_sweep(
#        X1_train, Y1_train, X1_test, Y1_test,
#        X2_train, Y2_train, X2_test, Y2_test,
#        param_name  = name,
#        param_values= values,
#        train_fn    = train_random_forest,
#        eval_fn     = evaluate_model,
#        labels      = labels
#
#    )
#
#    print("------------next training-------------")
#    plot_param_sweep(df, name, values, labels)


