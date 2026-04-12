import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./k-means/Mall_Customers.csv")

# remove unwanted features
df = df.drop(["Gender"], axis=1)

# removed rows of each columns whose values were 0
