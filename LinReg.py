import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



df= pd.read_csv("data.csv",sep=';',decimal=',')


X = df.iloc[:,:-1]
y = df.iloc[:,-1]

scaler = MinMaxScaler()
scaler.fit(X)
X= scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train,y_train)
accuracy = regressor.score(X_test,y_test)
print(accuracy)
y=np.array(y)
forecasts = regressor.predict(X).reshape(-1,1)

plt.plot(range(y.shape[0]),y)
plt.plot(range(forecasts.shape[0]),forecasts)
plt.show()




