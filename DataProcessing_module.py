'''
rewriting of the old "RandomForestModule.py", making it modular and easier to use
this part focuses on data preprocessing so we can:
1) load our data
2) create x sampled data from main data
3) create features|class separation for training
4) assign data to train|test split for RandomForest module
'''

import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(filepath):
    return pd.read_csv(filepath)

def get_balanced_subset(data, class_col='CLASS', samples_per_class=53):
    classes = sorted(data[class_col].unique())
    subsets = [data[data[class_col] == c].sample(n=samples_per_class, random_state=42) for c in classes]
    return pd.concat(subsets, axis=0).reset_index(drop=True)

def extract_features_and_labels(data):
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    return X, Y

# ---------------- MODULE: Data Assignment ----------------
def split_data(X, Y, test_size=0.2, random_state=42, shuffle=True):
    return train_test_split(X, Y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=Y)
