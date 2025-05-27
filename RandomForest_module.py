'''
rewriting of the old "RandomForestModule.py", making it modular and esier to use.
This part focuses on model creation:
1) creats RF modul using sklearn
2) evaluates our model on 6 difrenet parameters
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------- MODULE: Model Training ----------------
def train_random_forest(X_train, Y_train, n_estimators=40, max_depth=15, max_features=None, random_state=42):
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
    return df, accuracy