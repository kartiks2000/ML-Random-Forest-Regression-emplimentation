# RANDOM FOREST REGRESSION

# Importing liberaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Importing dataset
dataset = pd.read_csv("Position_Salaries.csv")



# Seperating dependent and independent variables
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values



# We dont need to split dataset in RANDOM FOREST REGRESSION 
# Splitting the dataset into Training set and Test set
'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
'''


# We dont need apply feature scaling in RANDOM FOREST REGRESSION
# Feature Scaling -> Normalizing the range of data/vairiable values
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
# We can also rescale y if we need
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)
'''




# Fitting RANDOM FOREST REGRESSION model to the dataset
from sklearn.ensemble import RandomForestRegressor
# We have to pass the number of Decision Trees we want in our forest, we can vary number of tress as per our need
# We pass "random_state" value so that our result and the instructor result in video is same.
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(x,y)




# Predicting a new result with RANDOM FOREST REGRESSION
y_pred = regressor.predict([[6.5]])
print(y_pred)





# This graph wouldn't show the correct graph because it can show non-linear and non contineous graph hence we used high resolution graph to plot
# Visualising the RANDOM FOREST REGRESSION results
'''
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (RANDOM FOREST REGRESSION)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''



# Visualising the RANDOM FOREST REGRESSION results (for higher resolution and smoother curve)
# If we want the graph to be more accurate by making inputs complicated.
# It gives a much smoother curve.
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (RANDOM FOREST REGRESSION)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
