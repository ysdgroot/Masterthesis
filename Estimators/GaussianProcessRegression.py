from sklearn import gaussian_process
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform
from sklearn import preprocessing
import modelsaver
import time
from Estimators import preprocessing_data as prep
import numpy as np

########################################################
# ---------------- PARAMETERS --------------------------#
########################################################
kernels = [RBF(), Matern(), DotProduct(), WhiteKernel(), RationalQuadratic()]

alpha = None  # (default=1e-10) adding to diagonal kernel matrix

normalize_y = [True, False]

param_grid = {"normalize_y": [True, False],
              'kernel': kernels,
              "n_restarts_optimizer": [i for i in range(10)],
              "alpha": uniform(loc=0.000000001, scale=0.001)}


########################################################################################################################


def cv_gpr_models(stockmodel, option, random_state=None, scale=False):
    param_grid = {"normalize_y": [True, False],
                  'kernel': kernels,
                  "alpha": uniform(loc=0.000000001, scale=0.001)}

    datamanager = prep.DataManager(model=stockmodel, column_fitting=option)
    X, y = datamanager.get_random_training_data(10000)

    if scale:
        scaler = preprocessing.StandardScaler().fit(X, y)
        X = scaler.transform(X)

    gpr = gaussian_process.GaussianProcessRegressor(optimizer="fmin_l_bfgs_b")
    clf = RandomizedSearchCV(gpr, param_grid, random_state=random_state, cv=3, n_iter=50, verbose=10, n_jobs=2,
                             scoring=['neg_mean_squared_error', 'r2'],
                             refit=False)

    performance = clf.fit(X, y)

    string_scaled = '_scaled' if scale else ""
    modelsaver.save_model(performance, f"GPR-random_search_{stockmodel}_{option}{string_scaled}_random{random_state}")


def main_cv():
    models = ["BS", "VG", "H"]
    columns_fitting = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    start_random_state = 594

    for i, model in enumerate(models):
        for j, option in enumerate(columns_fitting):
            print(f"Start cv for {model}-{option}")
            cv_gpr_models(model, option, random_state=start_random_state + 10 * i + j * 2, scale=True)

    print("End")


if __name__ == '__main__':
    print("Start")
    main_cv()
