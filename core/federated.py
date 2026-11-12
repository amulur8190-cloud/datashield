import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def init_weights(n_features):
    # all zeros to start; the last number is the "bias"
    return np.zeros(n_features + 1)


def predict_proba(w, X):
    return sigmoid(X @ w[:-1] + w[-1])


def class_weights(y):
    # fraud is rare, so each fraud counts more (same idea as "balanced")
    n = len(y)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(n - n_pos, 1)
    return n / (2 * n_neg), n / (2 * n_pos)  # (normal weight, fraud weight)


def local_train(w_global, X, y, epochs=3, lr=0.1, batch_size=512, seed=0):
    """One bank improves the global model using ONLY its own data."""
    w = w_global.copy()
    rng = np.random.default_rng(seed)
    w_neg, w_pos = class_weights(y)
    sample_w = np.where(y == 1, w_pos, w_neg)

    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            b = order[start:start + batch_size]
            p = predict_proba(w, X[b])
            err = (p - y[b]) * sample_w[b]
            w[:-1] -= lr * (X[b].T @ err) / len(b)
            w[-1] -= lr * err.mean()
    return w


def fed_avg(updates, sizes):
    """Server: average the banks' weights (banks with more data count more)."""
    sizes = np.array(sizes, dtype=float)
    return np.average(np.array(updates), axis=0, weights=sizes / sizes.sum())


def evaluate(w, X, y):
    p = predict_proba(w, X)
    pred = (p >= 0.5).astype(int)
    eps = 1e-12
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "loss": loss,
    }


def run_federated(banks, X_test, y_test, rounds=10):
    n_features = X_test.shape[1]
    w_global = init_weights(n_features)
    history = []

    for r in range(1, rounds + 1):
        updates, sizes = [], []
        for i, (name, (X, y)) in enumerate(banks.items()):
            w_local = local_train(w_global, X, y, seed=r * 10 + i)
            updates.append(w_local)  # only these numbers leave the bank
            sizes.append(len(y))

        w_global = fed_avg(updates, sizes)
        m = evaluate(w_global, X_test, y_test)
        m["round"] = r
        history.append(m)

    return w_global, history