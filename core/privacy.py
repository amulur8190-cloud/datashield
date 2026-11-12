import numpy as np
from core.federated import init_weights, local_train, fed_avg, evaluate


def clip_and_noise(w_global, w_local, clip, noise, rng):
    """Differential-privacy style protection: limit the update size, then add noise."""
    delta = w_local - w_global
    norm = np.linalg.norm(delta)
    if norm > clip:
        delta = delta * (clip / norm)
    if noise > 0:
        delta = delta + rng.normal(0, noise * clip, size=delta.shape)
    return w_global + delta


def run_private(banks, X_test, y_test, rounds=5, clip=1.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    w_global = init_weights(X_test.shape[1])
    history = []
    for r in range(1, rounds + 1):
        updates, sizes = [], []
        for i, (name, (X, y)) in enumerate(banks.items()):
            w_local = local_train(w_global, X, y, seed=r * 10 + i)
            updates.append(clip_and_noise(w_global, w_local, clip, noise, rng))
            sizes.append(len(y))
        w_global = fed_avg(updates, sizes)
        m = evaluate(w_global, X_test, y_test)
        m["round"] = r
        history.append(m)
    return w_global, history