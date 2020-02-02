import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import time

import pickle
import ModelsStock.BlackScholes as BS

########################################################
#---------------- PARAMETERS --------------------------#
########################################################
n_estimators = 100   # testen of meerdere beter/slechter zijn
criterion = ["mse", "mae"]
max_features = ["auto", "sqrt", "log2"]
warm_start = [True, False]
bootstrap = [True, False]


########################################################################################################################


np.random.seed(123)

data = BS.get_random_data_and_solutions('C', 10000, [80, 120], [0.01, 0.03], [0.01, 0.2], [0.25, 5], [70, 130])
print('End new data')

X = data.drop('Value_option', axis=1)
y = data['Value_option']

start = time.time()
model_random_forest = RandomForestRegressor().fit(X, y)
end = time.time()
print('Time: ' + str(end - start))

data_test = BS.get_random_data_and_solutions('C', 1000, [90, 110], [0.01, 0.03], [0.01, 0.2], [0.25, 5], [80, 120])
X_test = data_test.drop('Value_option', axis=1)
y_test = data_test['Value_option']


pred_y = model_random_forest.predict(X_test)


score = model_random_forest.score(X_test, y_test)
print(score)

mse = mean_squared_error(y_test, pred_y)
print("Mean Squared Error:", mse)

rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)
