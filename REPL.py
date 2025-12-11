import pandas as pd
df = pd.read_csv("creditcard1.csv")
print(df.shape)              # rows, cols
print(df['Class'].value_counts())  # class imbalance
print(df.describe())