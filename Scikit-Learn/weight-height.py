import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model, metrics


def convert_in_to_cm(x):  # convert inches to centimeter
    return x * 2.54


def convert_ibs_to_kg(y_predict):  # convert pound to kilogram

    return y_predict * 0.45359237


df = pd.read_csv("weight-height.csv")
df["Gender"] = df["Gender"].apply(lambda _: 1 if _ == "Male" else 0)
X = df.loc[:, ["Gender", "Height"]].values  # input dates
y = df["Weight"].values  # output dates
# print(y)

model = linear_model.LinearRegression()
model.fit(X, y)

mse = metrics.mean_squared_error(model.predict(X), y)
print(f"Mean squared error: {mse}")
print(f"Regression coefficient: {model.coef_}")
print(f"Error: {model.intercept_}")
print(f"Formula: Weight = {model.coef_[0]}*Gender + {model.coef_[1]}*Height + {model.intercept_})")

while True:
    x = float(input("Enter a height in inches: "))
    y_predict_female = model.predict([[0, x]])
    y_predict_male = model.predict([[1, x]])
    print("Weight prediction:")
    print(f"Female height {x} inches/ {convert_in_to_cm(x)} cm,"
          f" predictive weight: {y_predict_female[0]} pound/ {convert_ibs_to_kg(y_predict_female[0])} kg")
    print(f"Male height {x} inches/ {convert_in_to_cm(x)} cm,"
          f" predictive weight: {y_predict_male[0]} pound/ {convert_ibs_to_kg(y_predict_male[0])} kg")
