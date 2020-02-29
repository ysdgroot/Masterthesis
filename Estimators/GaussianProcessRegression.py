import numpy as np
import math
from sklearn import gaussian_process
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, DotProduct
import ModelSaver
import time


########################################################
#---------------- PARAMETERS --------------------------#
########################################################
kernel = [RBF, WhiteKernel, Matern, DotProduct, ConstantKernel]     #TODO: bekijk de specifieke parameters per kernel
alpha = None        # (default=1e-10) adding to diagonal kernel matrix

n_restarts_optimizer = [i for i in range(5)]
normalize_y = [True, False]

########################################################################################################################


# kernel = DotProduct()
kernel = RBF()
start = time.time()
model_gauss = gaussian_process.GaussianProcessRegressor(kernel=kernel).fit(X, y)
end = time.time()
print('Time: ' + str(end - start))

# pred_y = model_gauss.predict(X_test)
#
# score = model_gauss.score(X_test, y_test)
# print(score)
#
# mse = mean_squared_error(y_test, pred_y)
# print("Mean Squared Error:", mse)
#
# rmse = math.sqrt(mse)
# print("Root Mean Squared Error:", rmse)
