import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
print('---------------- DATA PREPARATION --------------')
dataset = pd.read_csv('weather.csv')
# print(dataset.head())
# print(dataset.shape)

X = dataset[['MinTemp', 'MaxTemp']] #'Evaporation',  'Sunshine'
y = dataset['Rainfall']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('---------------- TRAINING THE MODEL --------------')
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# with open('wx_model.pkl', 'wb') as file:
# 	pickle.dump(regressor, file)

#To retrieve the intercept:
print("Intercept: ")
print(regressor.intercept_)
print("\n ")
#For retrieving the slope:
print("Coef Slope: ")
print(regressor.coef_)
print("\n ")
print("Actual vs predicted result of test dataset: ")
y_pred = regressor.predict(X_test)
# print(y_pred)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.head())


print('------------------- PREDICT NEW DATA -----------------------------')
new_min_temp = 14
new_max_temp = 25
new_data = [[new_min_temp, new_max_temp]]
print('Predicted Rainfall: ', regressor.predict(new_data)[0])

print('------------------- DONE -----------------------------')