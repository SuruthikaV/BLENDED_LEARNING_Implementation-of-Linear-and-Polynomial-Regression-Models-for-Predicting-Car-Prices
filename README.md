# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and prepare data
2. Split data into train and test sets
3. Fit Linear Regression model using a Pipeline
4. Fit Polynomial Regression model (degree 2) using a Pipeline
5. Evaluate models and visualize results

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt

df=pd.read_csv('encoded_car_data (1).csv')
x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

lr=Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
lr.fit(x_train,y_train)
y_pred_linear=lr.predict(x_test)

poly_model=Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])
poly_model.fit(x_train,y_train)
y_pred_poly=poly_model.predict(x_test)

print('Name:Suruthika V ')
print('Reg. No:212225040441 ')
print("linear Regression:")

print('MSE=',mean_squared_error(y_test,y_pred_linear))
r2score=r2_score(y_test,y_pred_linear)
print('R2 Score=',r2score)
print('MAE=',mean_absolute_error(y_test,y_pred_linear))

print("\nPolynomial Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"R2: {r2_score(y_test,y_pred_poly):.2f}")

plt.figure(figsize=(10, 5))
plt.scatter(y_test,y_pred_linear,label='Linear',alpha=0.6)
plt.scatter(y_test,y_pred_poly,label='Polynomial (degree=2)',alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(), y.max()],'r--',label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: Suruthika V
RegisterNumber: 212225040441 
*/
```

## Output:
<img width="338" height="153" alt="Screenshot 2026-02-11 203104" src="https://github.com/user-attachments/assets/f8ab5be4-e45b-4bc6-a73b-bc6357628403" />
<img width="1161" height="715" alt="Screenshot 2026-02-11 203119" src="https://github.com/user-attachments/assets/b4203d7a-7dd6-4088-b722-4d9a1b563eb9" />




## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
