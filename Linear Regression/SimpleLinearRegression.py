# Importing necessary  libraries
# %matplotlib inline
import numpy as np
import pandas as pd

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib','inline')

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0,10.0)

# Reading data
data = pd.read_csv('headbrain.csv')
print(data.shape)
print(data.head())

# Fetching X and Y values
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

# Now we will find the coefficients b0 and b1
# To find coeff we need mean of x and Y

# Mean of X and Y
X_mean = np.mean(X)
Y_mean = np.mean(Y)

# total values
m = len(X)

# Using the formula to calculate b0 and b1
sigma_numer = 0
deno = 0
for i in range(m):
    sigma_numer += (X[i] - X_mean) * (Y[i] - Y_mean)
    deno += (X[i] - X_mean)**2
b1 = sigma_numer / deno
b0 = Y_mean - (b1 * X_mean)

# Coefficients
print(b1,b0)

# Plotting values and regression line
max_X = np.max(X) + 100
min_X = np.min(X) - 100

# Calculate line values X and Y
x = np.linspace(min_X,max_X,1000)
y = b0 + b1 * x

# Ploting line
plt.plot(x,y,color='Blue',label="Regression Line")
# Scatter Plot
plt.scatter(X,Y,c='Red',label="Scatter Plot")

plt.xlabel("Head Size in cm3")
plt.ylabel("Brain weight in grams")
plt.legend()
plt.show()

# Now we will evaluate the model, there are many methods, RMSE, Coefficient of Determination
# Calculating root mean squared error
rmse = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/m)
print(rmse)

# Determining R^2
# first we will calculate, total sum of squares(tss) and residual sum of squares(rss)

tss = 0
rss = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    tss += (Y[i] - Y_mean) ** 2
    rss += (Y[i] - y_pred) ** 2
r2 = 1 - (rss/tss)
print(r2)

# -------------------------------------------------
# Scikit-learn approach
# -------------------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = X.reshape((m,1))
# Creating model
reg = LinearRegression()
# Fitting training data
reg = reg.fit(X,Y)
# Y prediction
y_pred = reg.predict(X)

# Calculating RMSE and R2 score
mse = mean_squared_error(Y,y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(X,Y)

print(rmse)
print(r2_score)
