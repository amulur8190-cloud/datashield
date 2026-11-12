from core.data_loader import load_and_split
from core.federated import run_federated

banks, X_test, y_test = load_and_split()

w_global, history = run_federated(banks, X_test, y_test, rounds=10)

print("FEDERATED LEARNING (FedAvg)\n")
for h in history:
    print(f"Round {h['round']:>2}: precision={h['precision']:.3f}  "
          f"recall={h['recall']:.3f}  f1={h['f1']:.3f}  loss={h['loss']:.4f}")