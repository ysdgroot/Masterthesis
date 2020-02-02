import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import time
import pickle

import ModelsStock.BlackScholes as BS

########################################################
#---------------- PARAMETERS --------------------------#
########################################################
# SVR, linearSVR, NUSVR
# linearSVR -> larger datasets (notes on scikit-learn)
kernels = ["rbf", "linear", "poly", "rbf", "sigmoid"]
degree = [i for i in range(1, 11, 1)]   # only for the kernel "poly"

gamma = ["scale", "auto"]   # kernel coefficient for 'rbf', 'poly' and 'sigmoid'

C = 1               # float, TODO: take random value,
epsilon = 0.1       # float, TODO: take random value,

########################################################################################################################


np.random.seed(123)

data = BS.get_random_data_and_solutions('C', 10000, [80, 120], [0.01, 0.03], [0.01, 0.2], [0.25, 5], [70, 130])
print('End new data')

X = data.drop('Value_option', axis=1)
y = data['Value_option']


# Start creating of model
model = SVR(C=3.0, cache_size=200, coef0=0.0, degree=4, epsilon=0.1, gamma='auto',
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.0001, verbose=False)

start = time.time()
model_svr = model.fit(X,  y)
end = time.time()
print('Time: ' + str(end - start))

pickle.dump(model_svr, "SVR_rbf_4")

data_test = BS.get_random_data_and_solutions('C', 1000, [90, 110], [0.01, 0.03], [0.01, 0.2], [0.25, 5], [80, 120])
X_test = data_test.drop('Value_option', axis=1)
y_test = data_test['Value_option']


pred_y = model_svr.predict(X_test)

score = model_svr.score(X_test, y_test)
print(score)

mse = mean_squared_error(y_test, pred_y)
print("Mean Squared Error:", mse)

rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)

