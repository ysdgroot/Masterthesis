from sklearn.svm import SVR
from Estimators import random_search_svr
from Estimators import DataCollector as dc
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
import modelsaver
import importlib.resources as pkg_resources
import numpy as np

########################################################
# ---------------- PARAMETERS --------------------------#
########################################################
kernels = ["rbf", "poly", "sigmoid"]

distributions = dict(C=uniform(loc=1, scale=50),
                     kernel=["rbf", "linear", "poly", "sigmoid"],
                     degree=[2, 3, 4],
                     gamma=["scale", "auto"],
                     epsilon=uniform(0.01, 5))


########################################################################################################################


def cv_svr_models(stockmodel, option_type, random_state):
    """
    For the given stockmodel and option type do a 3-fold cross validation of 50 random parametersets.

    Saves all the cross validations in "SVR-random_search_{stockmodel}_{option_type}_scaled_random{random_state}"

    :param stockmodel: str, "BS", "VG" or "H"
    :param option_type: str, "opt_standard", "opt_asianmean", "opt_lookbackmin" or "opt_lookbackmax"
    :param random_state: int, for the randomstate
    """
    datamanager = dc.DataManager(stockmodel=stockmodel, option_type=option_type)
    X, y = datamanager.get_training_data()

    # het SVR gaat veel sneller en presteert veel beter als de data wordt herschaald
    scaler = preprocessing.StandardScaler().fit(X, y)
    X = scaler.transform(X)

    svr = SVR(cache_size=1000)
    clf = RandomizedSearchCV(svr, distributions, random_state=random_state, cv=3, n_iter=50, verbose=10, n_jobs=6,
                             scoring=['neg_mean_squared_error', 'r2'],
                             refit=False)

    performance = clf.fit(X, y)

    modelsaver.save_model(performance, f"SVR-random_search_{stockmodel}_{option_type}")


def main_cv():
    """
    Do the 3-fold cross validation of 50 random parametersets for every stockmodel and option.
     Also saves the cross validation results
    """
    models = ["BS", "VG", "H"]
    columns_fitting = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    start_random_state = 257

    # Also do it for the Black-Scholes formula
    model = "BS"
    option = "opt_exact_standard"
    print(f"Start cv for {model}-{option}")
    cv_svr_models(model, option, random_state=start_random_state + 30)

    for i, model in enumerate(models):
        for j, option in enumerate(columns_fitting):
            print(f"Start cv for {model}-{option}")
            cv_svr_models(model, option, random_state=start_random_state + 10 * i + j * 2)

    print("End")


def full_data_training(stockmodel, option_type, only_call=False, with_percentage=False):
    """
    print the results of the performance over the full dataset for the given stock stockmodel and option type

    :param stockmodel: str, "BS", "VG" or "H"
    :param option_type: str, "opt_standard", "opt_asianmean", "opt_lookbackmin" or "opt_lookbackmax"
    :param only_call: bool (default=False), if the dataset only contains the call options
    :param with_percentage: bool (default=False),
            if the dataset needs to contain the percentage of the stock price and the strike
    """
    base_file_name = "SVR-random_search_{0}_{1}_scaled.p".format(stockmodel, option_type)

    # get the best parameters from the cross validation
    full_file_name = pkg_resources.open_text(random_search_svr, base_file_name).name
    dict_cv_results = modelsaver.get_model(full_file_name).cv_results_
    best_position = np.where(dict_cv_results['rank_test_neg_mean_squared_error'] == 1)
    best_model_parameters = np.array(dict_cv_results['params'])[best_position][0]

    dm = dc.DataManager(stockmodel=stockmodel,
                        option_type=option_type,
                        only_call=only_call,
                        with_percent=with_percentage)
    X_train, y_train = dm.get_training_data()

    scaler = preprocessing.StandardScaler().fit(X_train, y_train)
    X_train = scaler.transform(X_train)

    svr_model = SVR(cache_size=2000,
                    C=best_model_parameters['C'],
                    degree=best_model_parameters['degree'],
                    epsilon=best_model_parameters['epsilon'],
                    gamma=best_model_parameters['gamma'],
                    kernel=best_model_parameters['kernel'])

    svr_model.fit(X_train, y_train)

    X_test, y_test = dm.get_test_data()
    X_test = scaler.transform(X_test)

    y_pred = svr_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred=y_pred)

    print(f"MSE: {mse:4.3f}")


def main_full_data_training(only_call=False, with_percentage=False):
    """
    print the results of the performance over the full dataset, for all stockmodels and option types.
    """
    models = ["BS", "VG", "H"]
    option_types = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    if only_call:
        print("Only for call options")

    model = "BS"
    option = "opt_exact_standard"
    full_data_training(model, option, only_call=only_call, with_percentage=with_percentage)

    for stockmodel in models:
        for option_type in option_types:
            print(f"{stockmodel}:{option_type}")
            full_data_training(stockmodel, option_type, only_call=only_call, with_percentage=with_percentage)


def part_dataset_like_gpr(stockmodel, option_type, only_call=False):
    """
   Do the testings with a smaller set of datapoints, the same as the test for the Gaussian Process Regressor
   Print the mse of the Test data and the part of the training data which are not used

   :param stockmodel: str, "BS", "VG" or "H"
   :param option_type: str, "opt_standard", "opt_asianmean", "opt_lookbackmin" or
   :param only_call: bool (default=False), if the dataset only contains the call options
   :param with_percentage: bool (default=False),
           if the dataset needs to contain the percentage of the stock price and the strike
   :param scale: bool (default=False), whenever to scale the data
   """
    n_samples = 10000
    random_state = 9943

    base_file_name = "SVR-random_search_{0}_{1}_scaled.p".format(stockmodel, option_type)

    # get the best parameters from the cross validation
    full_file_name = pkg_resources.open_text(random_search_svr, base_file_name).name
    dict_cv_results = modelsaver.get_model(full_file_name).cv_results_
    best_position = np.where(dict_cv_results['rank_test_neg_mean_squared_error'] == 1)
    best_model_parameters = np.array(dict_cv_results['params'])[best_position][0]

    # get the training and test data
    dm = dc.DataManager(stockmodel=stockmodel, option_type=option_type, only_call=only_call)
    X_train, y_train, x_not_selected, y_not_selected = dm.get_random_training_data(n_samples=n_samples,
                                                                                   random_state=random_state,
                                                                                   get_not_selected_data=True)

    scaler = preprocessing.StandardScaler().fit(X_train, y_train)
    X_train = scaler.transform(X_train)

    svr_model = SVR(cache_size=2000,
                    C=best_model_parameters['C'],
                    degree=best_model_parameters['degree'],
                    epsilon=best_model_parameters['epsilon'],
                    gamma=best_model_parameters['gamma'],
                    kernel=best_model_parameters['kernel'])

    svr_model.fit(X_train, y_train)

    X_test, y_test = dm.get_test_data()
    X_test = scaler.transform(X_test)
    x_not_selected = scaler.transform(x_not_selected)

    y_pred = svr_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred=y_pred)

    y_pred_not_selected = svr_model.predict(x_not_selected)
    mse_not_selected = mean_squared_error(y_not_selected, y_pred_not_selected)

    print(f"MSE(test data): {mse_test:4.3f}")
    print(f"MSE(not selected): {mse_not_selected:4.3f}")


def main_part_dataset_like_gpr(only_call=False):
    """
    Prints for every option type from every stockmodel the mse of the best parameters of the cross validation

    :param only_call: bool (default=False), whenever to use only the call options
    """
    models = ["BS", "VG", "H"]
    option_types = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    model = "BS"
    option = "opt_exact_standard"
    part_dataset_like_gpr(model, option, only_call=only_call)

    for stockmodel in models:
        for option_type in option_types:
            print(f"{stockmodel}:{option_type}")
            part_dataset_like_gpr(stockmodel, option_type, only_call=only_call)


def train_real_data(symbol="SPX"):
    kernel = "rbf"
    C = 50
    dm_real = dc.DataManagerRealData(symbol)

    X_train, y_train = dm_real.get_training_data()
    X_test, y_test = dm_real.get_test_data()

    train_index = X_train.sample(n=100000).index
    X_train, y_train = X_train.loc[train_index], y_train.loc[train_index]

    svr_model = SVR(cache_size=3000, kernel=kernel, C=C)

    svr_model.fit(X_train, y_train)

    y_pred = svr_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"{symbol} - MSE: {mse}")


def main_real_data():
    for symbol in ["SPX", "SPXPM", "SX5E"]:
        train_real_data(symbol)


if __name__ == '__main__':
    print("Start")
    # main_cv()
    # main_full_data_training(only_call=False)
    # print("   ")
    # print("Dataset like GPR")
    # main_part_dataset_like_gpr(only_call=False)
    main_real_data()
