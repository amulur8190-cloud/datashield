from core.data_loader import load_and_split
from core.attacks import run_secure

banks, X_test, y_test = load_and_split()

cases = [
    ("1. Normal (no attack)", False, False),
    ("2. Attack, NO defence", True, False),
    ("3. Attack, WITH defence", True, True),
]

for title, attack, defence in cases:
    _, hist, log = run_secure(banks, X_test, y_test, attack=attack, defence=defence)
    h = hist[-1]
    print(f"{title}: precision={h['precision']:.3f}  recall={h['recall']:.3f}  "
          f"f1={h['f1']:.3f}  loss={h['loss']:.4f}")
    if defence:
        blocked = [(e["round"], e["bank"]) for e in log if e["status"] == "BLOCKED"]
        print("   Blocked updates:", len(blocked), "->", sorted(set(b for _, b in blocked)))