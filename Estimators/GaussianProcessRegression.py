from sklearn import gaussian_process
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct
from sklearn import preprocessing
import modelsaver
import time
from Estimators import preprocessing_data as prep
import numpy as np

########################################################
# ---------------- PARAMETERS --------------------------#
########################################################
kernels = [RBF(), Matern(), DotProduct()]  # TODO: bekijk de specifieke parameters per kernel

alpha = None  # (default=1e-10) adding to diagonal kernel matrix

n_restarts_optimizer = [i for i in range(5)]
normalize_y = [True, False]

param_grid = {'normalize_y': [True, False],
              'kernel': kernels,
              "n_restarts_optimizer": [0, 1, 2, 3, 4]}

########################################################################################################################

datamanager = prep.DataManager(column_fitting="opt_exact_standard")
# datamanager = prep.DataManager(column_fitting="opt_asianmean")
X, y = datamanager.get_training_data()
del X["strike_price_percent"]

random_positions = np.random.randint(0, 50000, size=10000)

# make the data smaller
X = X.iloc[random_positions.tolist()]
y = y.iloc[random_positions.tolist()]

X = preprocessing.scale(X)

X_test, y_test = datamanager.get_test_data()
del X_test["strike_price_percent"]

X_test = preprocessing.scale(X_test)

for kernel in kernels:
    print(f"Busy kernel {kernel}")
    start = time.perf_counter()
    model_gauss = gaussian_process.GaussianProcessRegressor(kernel=kernel,
                                                            normalize_y=False,
                                                            n_restarts_optimizer=3,
                                                            optimizer="fmin_l_bfgs_b").fit(X, y)
    end = time.perf_counter()
    print(f"Time: {end - start}")

    score_eval_train = model_gauss.score(X, y)
    print(f"Evaluation {score_eval_train}")

    prediction = model_gauss.predict(X_test)
    print(mean_squared_error(prediction, y_test))

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
