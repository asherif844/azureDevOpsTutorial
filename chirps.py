import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

url = 'https://raw.githubusercontent.com/Microsoft/AzureNotebooks/master/Samples/Linear%20Regression%20-%20Cricket%20Chirps/cricket_chirps.csv'


df = pd.read_csv(url)

df.to_csv('data/birds_chirping.csv')

x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=12345)


lr = LinearRegression()

lr.fit(x_train, y_train)

y_predict = lr.predict(x_test)

# print(y_test)
# print(y_predict)


plt.scatter(x_train, y_train, color='green', label = 'training data')
plt.scatter(x_test, y_test, color='red', label = 'testing data')
plt.scatter(x_test, y_predict, color='blue', label = 'predicted values')
plt.plot(x_train, lr.predict(x_train), color='grey', label = 'regression')

plt.title('Temperature vs Chirps/Minute')
plt.xlabel('Chirps/Minute')
plt.ylabel('Temperature (F)')
plt.legend()
plt.show()
