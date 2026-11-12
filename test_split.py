from core.data_loader import load_and_split

banks, X_test, y_test = load_and_split()

for name, (X, y) in banks.items():
    print(name, "-> transactions:", len(y), "| frauds:", int(y.sum()))

print("Shared test set ->", len(y_test), "transactions,", int(y_test.sum()), "frauds")