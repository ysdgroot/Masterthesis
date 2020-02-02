import numpy as np
import math
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import time

import ModelsStock.BlackScholes as BS

########################################################
#---------------- PARAMETERS --------------------------#
########################################################
Base_estimator = None    # decide which estimator to use (maybe the best of each Regressor)
n_estimators = 50        # number of estimators, TODO: take random values
learning_rate = 1        # shrinkage, trade_off between n_estimators, TODO: take random values
loss = ['linear', 'square', 'exponential']      # loss function for update weights

########################################################################################################################


np.random.seed(123)

data = BS.get_random_data_and_solutions('C', 10000, [80, 120], [0.01, 0.03], [0.01, 0.2], [0.25, 5], [70, 130])
print('End new data')

X = data.drop('Value_option', axis=1)
y = data['Value_option']


# model_for_ada = SVR(C=3.0, cache_size=200, coef0=0.0, degree=4, epsilon=0.1, gamma='auto',
#  kernel='rbf', max_iter=-1, shrinking=True, tol=0.0001, verbose=False)
# met SVR super slecht (duurt lang en heeft 87%)

model_for_ada = DecisionTreeRegressor()

start = time.time()
model_adaboost = AdaBoostRegressor(model_for_ada).fit(X, y)
end = time.time()
print('Time: ' + str(end - start))

data_test = BS.get_random_data_and_solutions('C', 1000, [90, 110], [0.01, 0.03], [0.01, 0.2], [0.25, 5], [80, 120])
X_test = data_test.drop('Value_option', axis=1)
y_test = data_test['Value_option']


pred_y = model_adaboost.predict(X_test)


score = model_adaboost.score(X_test, y_test)
print(score)

mse = mean_squared_error(y_test, pred_y)
print("Mean Squared Error:", mse)

rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)
