"""
Weight prediction based on gender and height in available data
"""

import pandas as pd
from sklearn import linear_model, metrics


def convert_in_to_cm(x_input):  # convert inches to centimeter
    return x_input * 2.54


def convert_ibs_to_kg(y_predict):  # convert pound to kilogram
    return y_predict * 0.45359237


# Get Data
df = pd.read_csv("weight-height.csv")
df["Gender"] = df["Gender"].apply(lambda _: 1 if _ == "Male" else 0)  # convert string to int
X = df.loc[:, ["Gender", "Height"]].values  # input date
y = df["Weight"].values  # output date

# Model
model = linear_model.LinearRegression()  # Model type
model.fit(X, y)  # Data training

# Model's information
mse = metrics.mean_squared_error(model.predict(X), y)
print(f"Mean squared error: {mse}")
print(f"Regression coefficient: {model.coef_}")
print(f"Error: {model.intercept_}")
print(f"Formula: Weight = {model.coef_[0]}*Gender + {model.coef_[1]}*Height + {model.intercept_})")

# Prediction
while True:
    x = float(input("Enter a height in inches: "))
    y_predict_female = model.predict([[0, x]])
    y_predict_male = model.predict([[1, x]])
    print("Weight prediction:")
    print(f"Female height: {x} inches / {convert_in_to_cm(x)} cm,"
          f" predictive weight: {y_predict_female[0]} pound / {convert_ibs_to_kg(y_predict_female[0])} kg")
    print(f"Male height: {x} inches / {convert_in_to_cm(x)} cm,"
          f" predictive weight: {y_predict_male[0]} pound / {convert_ibs_to_kg(y_predict_male[0])} kg")
