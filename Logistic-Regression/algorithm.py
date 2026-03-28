import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./Logistic-Regression/diabetes-dataset.csv")

# remove unwanted features
df = df.drop(["Pregnancies","DiabetesPedigreeFunction","BMI"], axis=1)

# removed rows of each columns whose values were 0
df = df[df["Glucose"] != 0]
df = df[df["BloodPressure"] != 0]
df = df[df["SkinThickness"] != 0]

# df[df['Glucose'] != 0] — the inner part df['Glucose'] != 0 returns True/False for each row. Then df[...] keeps only the rows where it's True. So you're basically saying "give me only rows where Glucose is not zero.


train_X = df.iloc[:373, :-1]
train_Y = df.iloc[:373,-1]

test_X = df.iloc[373:,:-1]
test_Y = df.iloc[373:,-1]

# feature scaling on training data
train_X["Glucose"] = (train_X["Glucose"] - train_X["Glucose"].mean()) / train_X["Glucose"].std()
train_X["BloodPressure"] = (train_X["BloodPressure"] - train_X["BloodPressure"].mean()) / train_X["BloodPressure"].std()
train_X["SkinThickness"] = (train_X["SkinThickness"] - train_X["SkinThickness"].mean()) / train_X["SkinThickness"].std()
train_X["Insulin"] = (train_X["Insulin"] - train_X["Insulin"].mean()) / train_X["Insulin"].std()
train_X["Age"] = (train_X["Age"] - train_X["Age"].mean()) / train_X["Age"].std()

# feature scaling on testing data
# Note - we scale testing data using mean and std of training data
test_X["Glucose"] = (test_X["Glucose"] - train_X["Glucose"].mean()) / train_X["Glucose"].std()
test_X["BloodPressure"] = (test_X["BloodPressure"] - train_X["BloodPressure"].mean()) / train_X["BloodPressure"].std()
test_X["SkinThickness"] = (test_X["SkinThickness"] - train_X["SkinThickness"].mean()) / train_X["SkinThickness"].std()
test_X["Insulin"] = (test_X["Insulin"] - train_X["Insulin"].mean()) / train_X["Insulin"].std()
test_X["Age"] = (test_X["Age"] - train_X["Age"].mean()) / train_X["Age"].std()


# Training
epoch = 2000
learning_rate = 0.01
m1 = 0
m2 = 0
m3 = 0
m4 = 0
m5 = 0
b = 0

loss_array = []

for i in range(epoch):

    # calculate linear prediction
    z = (m1 * train_X["Glucose"]) + (m2 * train_X["BloodPressure"]) + (m3 * train_X["SkinThickness"]) + (m4 * train_X["Insulin"]) + (m5 * train_X["Age"]) + b

    # feed sigmoid function which transforms it into value between 0 and 1
    prediction = 1 / (1 + np.exp(-z))

    # calculate loss using loss function
    loss = (-1/len(prediction)) * sum( (train_Y * (np.log(prediction))) + ((1 - train_Y) * (np.log(1 - prediction))))

    # append loss in loss array
    loss_array.append(loss)

    # gradient calculation
    grad_of_m1 = (1 / len(prediction)) * sum((prediction - train_Y) * train_X["Glucose"])
    grad_of_m2 = (1 / len(prediction)) * sum((prediction - train_Y) * train_X["BloodPressure"])
    grad_of_m3 = (1 / len(prediction)) * sum((prediction - train_Y) * train_X["SkinThickness"])
    grad_of_m4 = (1 / len(prediction)) * sum((prediction - train_Y) * train_X["Insulin"])
    grad_of_m5 = (1 / len(prediction)) * sum((prediction - train_Y) * train_X["Age"])

    grad_of_b = (1 / len(prediction)) * sum(prediction - train_Y)

    # update parameters
    m1 = m1 - (learning_rate * grad_of_m1)
    m2 = m2 - (learning_rate * grad_of_m2)
    m3 = m3 - (learning_rate * grad_of_m3)
    m4 = m4 - (learning_rate * grad_of_m4)
    m5 = m5 - (learning_rate * grad_of_m5)
    b = b - (learning_rate) * grad_of_b

print(loss_array[-6:])

# plot graph of loss curve against iterations/epoch. so we know, how loss changed as iterations occured

plt.plot(loss_array)
# This is the key line. You're only passing one list — loss_array. When you give plt.plot() a single list, matplotlib automatically treats the index of each element as the x value and the element's value as the y value  
# That's why you don't need to manually pass x values. The index IS the iteration number.

plt.xlabel("Epochs")   # just labels the x axis, purely visual
plt.ylabel("Loss")     # just labels the y axis, purely visual
plt.title("Training Loss Curve")  # title of the graph, purely visual
plt.show()             # actually renders and displays the graph