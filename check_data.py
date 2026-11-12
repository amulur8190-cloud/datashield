import pandas as pd

df = pd.read_csv("data/creditcard.csv")

print("Rows and columns:", df.shape)
print("Fraud count:", df["Class"].sum())
print("Fraud percentage:", round(df["Class"].mean() * 100, 3), "%")