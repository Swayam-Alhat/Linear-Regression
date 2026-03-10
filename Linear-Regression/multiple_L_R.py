import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./Linear-Regression/House_dataset.csv")

# select important features
df = df[["area","bedrooms","price"]]

# shuffle data So we get different data in both training & testing dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# check if null value is present
is_null = df.isnull().any().any()
print(is_null) # False
