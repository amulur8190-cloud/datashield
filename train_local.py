from core.data_loader import load_and_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

banks, X_test, y_test = load_and_split()

print("LOCAL MODELS (each bank trained alone)\n")

for name, (X, y) in banks.items():
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)  # the bank trains ONLY on its own data

    pred = model.predict(X_test)  # tested on the shared test set
    p = precision_score(y_test, pred, zero_division=0)
    r = recall_score(y_test, pred)
    f = f1_score(y_test, pred)

    print(f"{name}: precision={p:.3f}  recall={r:.3f}  f1={f:.3f}")