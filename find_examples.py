import numpy as np
from core.data_loader import load_and_split
from core.attacks import run_secure
from core.federated import predict_proba

banks, X_test, y_test = load_and_split()
w, _, _ = run_secure(banks, X_test, y_test, attack=False, defence=True)

prob = predict_proba(w, X_test)
fraud_idx = np.where(y_test == 1)[0]
normal_idx = np.where(y_test == 0)[0]

print("\nFRAUD transactions (type these numbers in the app):")
for i in fraud_idx[:6]:
    print(f"  #{i}  fraud probability = {prob[i]*100:.1f}%")

print("\nNORMAL transactions:")
for i in normal_idx[:6]:
    print(f"  #{i}  fraud probability = {prob[i]*100:.1f}%")