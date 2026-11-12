import numpy as np
from core.federated import init_weights, local_train, fed_avg, evaluate


def poison_update(w_global, w_local, scale=5.0):
    """ATTACK: the cheating bank flips its update and makes it 5x bigger."""
    delta = w_local - w_global
    return w_global - scale * delta


def detect_outliers(w_global, updates, factor=3.0):
    """DEFENCE: find updates that look very different from the others."""
    deltas = np.array([u - w_global for u in updates])
    middle = np.median(deltas, axis=0)               # the "typical" update
    dists = np.linalg.norm(deltas - middle, axis=1)  # how far each bank is from typical
    ref = max(np.median(dists), 1e-9)
    flagged = [bool(d > factor * ref) for d in dists]
    return flagged, dists


def run_secure(banks, X_test, y_test, rounds=10,
               attack=False, defence=False, malicious="Bank Gamma"):
    w_global = init_weights(X_test.shape[1])
    history, security_log = [], []

    for r in range(1, rounds + 1):
        names, updates, sizes = [], [], []
        for i, (name, (X, y)) in enumerate(banks.items()):
            w_local = local_train(w_global, X, y, seed=r * 10 + i)
            if attack and name == malicious:
                w_local = poison_update(w_global, w_local)
            names.append(name)
            updates.append(w_local)
            sizes.append(len(y))

        flagged = [False] * len(names)
        if defence:
            flagged, dists = detect_outliers(w_global, updates)
            for n, d, f in zip(names, dists, flagged):
                security_log.append({"round": r, "bank": n,
                                     "distance": float(d),
                                     "status": "BLOCKED" if f else "Trusted"})

        keep = [i for i, f in enumerate(flagged) if not f]
        if not keep:  # safety: never aggregate nothing
            keep = list(range(len(names)))
        w_global = fed_avg([updates[i] for i in keep], [sizes[i] for i in keep])

        m = evaluate(w_global, X_test, y_test)
        m["round"] = r
        history.append(m)

    return w_global, history, security_log