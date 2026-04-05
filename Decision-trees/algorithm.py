import pandas as pd
import numpy as np

df = pd.read_csv("./Logistic-Regression/diabetes-dataset.csv")

# remove unwanted features
df = df.drop(["Pregnancies","DiabetesPedigreeFunction","BMI"], axis=1)

# removed rows of each columns whose values were 0
df = df[df["Glucose"] != 0]
df = df[df["BloodPressure"] != 0]
df = df[df["SkinThickness"] != 0]

# df[df['Glucose'] != 0] — the inner part df['Glucose'] != 0 returns True/False for each row. Then df[...] keeps only the rows where it's True. So you're basically saying "give me only rows where Glucose is not zero.

# training data
train_X = df.iloc[:373, :-1]
train_Y = df.iloc[:373,-1]

# testing data
test_X = df.iloc[373:,:-1]
test_Y = df.iloc[373:,-1]


# implementation 

# split feature in such a way that it creates two groups
f1_unique_values = np.sort(train_X["Glucose"].unique()) 
f2_unique_values = np.sort(train_X["BloodPressure"].unique()) 
f3_unique_values = np.sort(train_X["SkinThickness"].unique())
f4_unique_values = np.sort(train_X["Insulin"].unique())
f5_unique_values = np.sort(train_X["Age"].unique())

split_values_for_f1 = (f1_unique_values[:-1] + f1_unique_values[1:]) / 2
split_values_for_f2 = (f2_unique_values[:-1] + f2_unique_values[1:]) / 2
split_values_for_f3 = (f3_unique_values[:-1] + f3_unique_values[1:]) / 2
split_values_for_f4 = (f4_unique_values[:-1] + f4_unique_values[1:]) / 2
split_values_for_f5 = (f5_unique_values[:-1] + f5_unique_values[1:]) / 2



