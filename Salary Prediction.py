import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#Load the dataSet 
dataset="C:\\Users\\Lenovo\\Desktop\\ML_Implementation\\Linear Regression\\Salary_Data_Set.xlsx"
data=pd.read_excel(dataset)
print(data)
#check First few rows
print(data.head())
#Define Features(X)and Target(Y)
X=data[['YearsExperience']]  #Independent Variable
Y=data['Salary']   #Dependent Variable
#Split into Training and Testing Sets
X_train,X_test,Y_train,Y_test =train_test_split(
    X,Y,test_size=0.2, random_state=42)
#train the Linear Regression Model 
model =LinearRegression()
model.fit(X_train,Y_train)
#Make Predictions
Y_pred =model.predict(X_test)
#comparre actual vs predicted
comparison =pd.DataFrame({'Actual':Y_test, 'predicted':Y_pred})
print(comparison)
#Visualize The Regression Lune 
plt.scatter(X,Y,color='blue',label='Actual Data')
plt.plot(X,model.predict(X),color='red',linewidth=2, label='Regression Line')
plt.xlabel("YearExperience")
plt.ylabel("Salary")
plt.title("Salary Prediction Using Linear Regression")
plt.legend()
plt.show()
#Evaluate the Model 
from sklearn.metrics import mean_squared_error, r2_score

print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred))
print("R² Score:", r2_score(Y_test, Y_pred))

