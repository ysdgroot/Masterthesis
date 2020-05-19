from sklearn import gaussian_process
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, RationalQuadratic
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn import preprocessing
import modelsaver
from Estimators import DataCollector as dc
from Estimators import random_search_gpr
import importlib.resources as pkg_resources
import numpy as np

########################################################
# ---------------- PARAMETERS --------------------------#
########################################################
kernels = [RBF(), Matern(), DotProduct(), RationalQuadratic()]

alpha = None  # (default=1e-10) adding to diagonal kernel matrix

param_grid = {"normalize_y": [True, False],
              'kernel': kernels,
              "n_restarts_optimizer": [i for i in range(10)],
              "alpha": uniform(loc=0.000000001, scale=0.001)}


########################################################################################################################


def cv_gpr_models(stockmodel, option, random_state=None, scale=False):
    """
    For the given stockmodel and option type do a 3-fold cross validation of 50 random parametersets.

    Saves all the cross validations in f"GPR-random_search_{stockmodel}_{option}{string_scaled}_random{random_state}"

    :param stockmodel: str, "BS", "VG" or "H"
    :param option_type: str, "opt_standard", "opt_asianmean", "opt_lookbackmin" or "opt_lookbackmax"
    :param random_state: int, for the randomstate
    """
    kernels = [RBF(), Matern(), DotProduct(), RationalQuadratic()]
    param_grid = {"normalize_y": [True, False],
                  'kernel': kernels,
                  "alpha": uniform(loc=0.000000001, scale=0.001)}

    datamanager = dc.DataManager(stockmodel=stockmodel, option_type=option)
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
    modelsaver.save_model(performance, f"GPR-random_search_{stockmodel}_{option}{string_scaled}")


def main_cv():
    """
    Do the 3-fold cross validation of 50 random parametersets for every stockmodel and option.
    Also saves the cross validation results
    """
    models = ["BS", "VG", "H"]
    columns_fitting = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    start_random_state = 594

    model = "BS"
    option = "opt_exact_standard"
    print(f"Start cv for {model}-{option}")
    cv_gpr_models(model, option, random_state=start_random_state + 30, scale=True)

    for i, model in enumerate(models):
        for j, option in enumerate(columns_fitting):
            print(f"Start cv for {model}-{option}")
            cv_gpr_models(model, option, random_state=start_random_state + 10 * i + j * 2, scale=True)

    print("End")


def full_data_training(stockmodel, option_type, only_call=False, with_percentage=False):
    """
    print the results of the performance over the part of the dataset(*) for the given stock stockmodel and option type

    (*) hardware problems when full dataset is given.

    :param stockmodel: str, "BS", "VG" or "H"
    :param option_type: str, "opt_standard", "opt_asianmean", "opt_lookbackmin" or "opt_lookbackmax"
    :param only_call: bool (default=False), if the dataset only contains the call options
    :param with_percentage: bool (default=False),
            if the dataset needs to contain the percentage of the stock price and the strike
    """
    n_samples = 10000
    random_state = 9943

    base_file_name = "GPR-random_search_{0}_{1}_scaled.p".format(stockmodel, option_type)

    full_file_name = pkg_resources.open_text(random_search_gpr, base_file_name).name
    dict_cv_results = modelsaver.get_model(full_file_name).cv_results_
    best_position = np.where(dict_cv_results['rank_test_neg_mean_squared_error'] == 1)
    best_model_parameters = np.array(dict_cv_results['params'])[best_position][0]

    dm = dc.DataManager(stockmodel=stockmodel,
                        option_type=option_type,
                        only_call=only_call,
                        with_percent=with_percentage)
    X_train, y_train, x_not_selected, y_not_selected = dm.get_random_training_data(n_samples=n_samples,
                                                                                   random_state=random_state,
                                                                                   get_not_selected_data=True)

    scaler = preprocessing.StandardScaler().fit(X_train, y_train)
    X_train = scaler.transform(X_train)

    gpr_model = gaussian_process.GaussianProcessRegressor(kernel=best_model_parameters["kernel"],
                                                          normalize_y=best_model_parameters["normalize_y"],
                                                          alpha=best_model_parameters["alpha"])

    gpr_model.fit(X_train, y_train)

    X_test, y_test = dm.get_test_data()
    X_test = scaler.transform(X_test)
    x_not_selected = scaler.transform(x_not_selected)

    y_pred = gpr_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred=y_pred)

    y_pred_not_selected = gpr_model.predict(x_not_selected)
    mse_not_selected = mean_squared_error(y_not_selected, y_pred_not_selected)

    print(f"MSE(test data): {mse_test}")
    print(f"MSE(not selected): {mse_not_selected}")


def main_full_data_training(only_call=False):
    """
    print the results of the performance over the full dataset, for all stockmodels and option types.
    """
    models = ["BS", "VG", "H"]
    option_types = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    if only_call:
        print("Only call options")

    model = "BS"
    option = "opt_exact_standard"
    full_data_training(model, option, only_call=only_call)

    for stockmodel in models:
        for option_type in option_types:
            print(f"{stockmodel}:{option_type}")
            full_data_training(stockmodel, option_type, only_call=only_call)


def main_real_data():
    kernel = Matern()
    for symbol in ["SPX", "SPXPM", "SX5E"]:
        dm_real = dc.DataManagerRealData(symbol)

        X_train, y_train = dm_real.get_training_data()

        train_index = X_train.sample(n=5000).index
        X_train, y_train = X_train.loc[train_index], y_train.loc[train_index]

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)

        gpr_model = gaussian_process.GaussianProcessRegressor(kernel=kernel)

        gpr_model.fit(X_train, y_train)

        X_test, y_test = dm_real.get_test_data()
        X_test = scaler.transform(X_test)

        y_pred = gpr_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print(f"{symbol} - MSE: {mse}")


if __name__ == '__main__':
    print("Start")
    # main_full_data_training(only_call=False)
    main_real_data()
    # main_cv()
