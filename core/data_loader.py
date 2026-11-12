import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/creditcard.csv"
BANK_NAMES = ["Bank Alpha", "Bank Beta", "Bank Gamma"]


def load_and_split(seed=42):
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class", "Time"]).values
    y = df["Class"].values

    # Keep a global test set that no bank ever trains on
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # Scale features (fit on training data only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Shuffle the training data and divide it into 3 banks
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_train))
    parts = np.array_split(idx, 3)

    banks = {}
    for name, p in zip(BANK_NAMES, parts):
        banks[name] = (X_train[p], y_train[p])

    return banks, X_test, y_test