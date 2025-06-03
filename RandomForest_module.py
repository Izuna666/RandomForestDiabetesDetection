'''
rewriting of the old "RandomForestModule.py", making it modular and esier to use.
This part focuses on model creation and modification as we as veryfication:
1) creats RF modul using sklearn
2) evaluates our model on 6 difrenet parameters
3) trains and evaluates Random Forest models over a sweep of one hyperparameter.
4) Plots metrics over a hyperparameter sweep.
5) Displays side-by-side confusion matrices for two sets of true/predicted labels.
'''

import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

# ---------------- MODULE: Model Training ----------------
def train_random_forest(X_train, Y_train, n_estimators=40, max_depth=None, max_features='sqrt', random_state=42):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 max_features=max_features, random_state=random_state)
    clf.fit(X_train, Y_train)
    return clf

# ---------------- MODULE: Evaluation ----------------
def evaluate_model(Y_test, Y_pred, labels):
    results = []
    for label in labels:
        precision, recall, f_score, support = precision_recall_fscore_support(
            Y_test == label, Y_pred == label, zero_division=0
        )
        results.append([
            label,
            recall[0],  # specificity
            recall[1],
            precision[1],
            f_score[1],
            support[1]
        ])
    df = pd.DataFrame(results, columns=["label", "specificity", "recall", "precision", "f_score", "support"])
    accuracy = accuracy_score(Y_test, Y_pred)
    #print(df)
    return df, accuracy


def evaluate_param_sweep(
    X1_train: pd.DataFrame,
    Y1_train: pd.Series,
    X1_test: pd.DataFrame,
    Y1_test: pd.Series,
    X2_train: pd.DataFrame,
    Y2_train: pd.Series,
    X2_test: pd.DataFrame,
    Y2_test: pd.Series,
    param_name: str,
    param_values: list,
    train_fn,
    eval_fn,
    labels: list
) -> pd.DataFrame:
    """
    Trains and evaluates Random Forest models over a sweep of one hyperparameter.

    Parameters:
    - X1_train, Y1_train, X1_test, Y1_test: full dataset splits
    - X2_train, Y2_train, X2_test, Y2_test: subset dataset splits
    - param_name: name of sklearn RF parameter to vary (e.g., 'max_features', 'n_estimators', 'max_depth')
    - param_values: list of values for that parameter
    - train_fn: function(X_train, Y_train, **kwargs) -> trained model
    - eval_fn: function(Y_true, Y_pred, labels) -> (df_metrics, accuracy)
    - labels: list of class labels for evaluation

    Returns:
    - DataFrame with columns [label, specificity, recall, precision, f_score, accuracy, model, param, value]
    """
    all_results = []
    for value in param_values:
        start = time.time()
        kwargs = {param_name: value}

        # Train
        clf_full = train_fn(X1_train, Y1_train, **kwargs)
        clf_sub = train_fn(X2_train, Y2_train, **kwargs)

        # Predict
        y_full_pred = clf_full.predict(X1_test)
        y_sub_pred = clf_sub.predict(X2_test)

        # Evaluate
        df_full, acc_full = eval_fn(Y1_test, y_full_pred, labels)
        df_sub, acc_sub = eval_fn(Y2_test, y_sub_pred, labels)

        print(f"{param_name}={value}: Full acc={acc_full*100:.2f}%, 53Samples acc={acc_sub*100:.2f}%, time={time.time()-start:.2f}s")

        # Tag
        df_full = df_full.assign(
            accuracy=acc_full,
            model='All samples',
            param=param_name,
            value=value
        )
        df_sub = df_sub.assign(
            accuracy=acc_sub,
            model='53 samples',
            param=param_name,
            value=value
        )
        all_results.append(pd.concat([df_full, df_sub], ignore_index=True))

    return pd.concat(all_results, ignore_index=True)


def plot_param_sweep(
    df: pd.DataFrame,
    param_name: str,
    param_values: list,
    labels: list,
    metrics: list = ['specificity', 'recall', 'precision', 'f_score']
) -> None:
    """
    Plots metrics over a hyperparameter sweep.

    Parameters:
    - df: DataFrame from evaluate_param_sweep
    - param_name: hyperparameter name
    - param_values: list of param values
    - labels: class labels
    - metrics: list of metric names to plot
    """
    # Map values to numeric positions
    x_pos = list(range(len(param_values)))
    val_to_pos = {v: i for i, v in enumerate(param_values)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    models = df['model'].unique()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for lbl in labels:
            for mdl in models:
                subset = df[(df['label'] == lbl) & (df['model'] == mdl)]
                positions = [val_to_pos[v] for v in subset['value']]
                ax.plot(
                    positions,
                    subset[metric],
                    marker='o',
                    label=f"{mdl} - Class {lbl}",
                    linewidth=2.5,
                    alpha=0.8
                )
        ax.set_title(metric.capitalize())
        ax.set_xlabel(param_name)
        ax.set_ylabel('Value')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(v) for v in param_values], rotation=45)
        ax.grid(True, axis='x')
        ax.legend()

    plt.suptitle(f"Performance vs. {param_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_confusion_matrices(
    Y1_test,
    Y1_pred,
    Y2_test,
    Y2_pred,
    titles=('Full Data classification', '53 Samples classification')
) -> None:
    """
    Displays side-by-side confusion matrices for two sets of true/predicted labels.

    Parameters:
    - Y1_test, Y1_pred: true and predicted for full dataset
    - Y2_test, Y2_pred: true and predicted for subset dataset (e.g., 53 samples)
    - titles: tuple of titles for the two plots

    Returns:
    - None (plots matrices)
    """
    # Compute confusion matrices
    cm1 = confusion_matrix(Y1_test, Y1_pred)
    cm2 = confusion_matrix(Y2_test, Y2_pred)

    # Plot side-by-side
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1)
    disp1.plot(ax=axes[0], colorbar=False)
    axes[0].set_title(titles[0])

    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
    disp2.plot(ax=axes[1], colorbar=False)
    axes[1].set_title(titles[1])

    plt.tight_layout()
    plt.show()