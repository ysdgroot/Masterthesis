import numpy as np
import math
from sklearn import gaussian_process
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, DotProduct

import time
import pickle
import ModelsStock.BlackScholes as BS

########################################################
#---------------- PARAMETERS --------------------------#
########################################################
kernel = [RBF, WhiteKernel, Matern, DotProduct, ConstantKernel]     #TODO: bekijk de specifieke parameters per kernel
alpha = None        # (default=1e-10) adding to diagonal kernel matrix

n_restarts_optimizer = [i for i in range(5)]
normalize_y = [True, False]

########################################################################################################################



np.random.seed(123)

data = BS.get_random_data_and_solutions('C', 10000, [80, 120], [0.01, 0.03], [0.01, 0.2], [0.25, 5], [70, 130])
print('End new data')

X = data.drop('Value_option', axis=1)
y = data['Value_option']

# kernel = DotProduct()
kernel = RBF()
start = time.time()
model_gauss = gaussian_process.GaussianProcessRegressor(kernel=kernel).fit(X, y)
end = time.time()
print('Time: ' + str(end - start))

data_test = BS.get_random_data_and_solutions('C', 1000, [90, 110], [0.01, 0.03], [0.01, 0.2], [0.25, 5], [80, 120])
X_test = data_test.drop('Value_option', axis=1)
y_test = data_test['Value_option']

pred_y = model_gauss.predict(X_test)

score = model_gauss.score(X_test, y_test)
print(score)

mse = mean_squared_error(y_test, pred_y)
print("Mean Squared Error:", mse)

rmse = math.sqrt(mse)
print("Root Mean Squared Error:", rmse)


