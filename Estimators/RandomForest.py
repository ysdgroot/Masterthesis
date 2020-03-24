import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV
from Estimators import preprocessing_data as prep
from scipy.stats import randint
import matplotlib.pyplot as plt
import time
from pathlib import Path
import modelsaver
from sklearn import preprocessing

########################################################
# ---------------- PARAMETERS -------------------------#
########################################################

warm_start = [True, False]
bootstap = [True, False]
n_estimators = [i for i in range(10, 601, 10)]


########################################################################################################################


# ["opt_exact_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]


# todo: bekijken voor het herschalen van de data - preprocessing

# Test for the MSE of the Randomforest regressor in function of the number of estimators
def do_graph_estimators(list_n_estimators, save_values=False):
    datamanager = prep.DataManager(column_fitting="opt_exact_standard")

    X, y = datamanager.get_training_data()
    del X["strike_price_percent"]

    test_data_X, test_data_y = datamanager.get_test_data()
    del test_data_X["strike_price_percent"]

    dict_mse_RF = {f'b{True}-w{True}': [], f'b{True}-w{False}': [], f'b{False}-w{True}': [], f'b{False}-w{False}': []}

    warm_starts = [True, False]
    bootstraps = [True, False]

    for i, n_estimator in enumerate(list_n_estimators):
        for warm_start in warm_starts:
            for bootstrap in bootstraps:
                random_forest_m1 = RandomForestRegressor(criterion="mse",
                                                         max_features="auto",
                                                         n_jobs=6,
                                                         n_estimators=n_estimator,
                                                         warm_start=warm_start,
                                                         bootstrap=bootstrap,
                                                         verbose=1,
                                                         random_state=113 + i + 2 * warm_start + 5 * bootstrap)

                random_forest_m1.fit(X, y)

                predictions = random_forest_m1.predict(test_data_X)

                mse_rf = mean_squared_error(test_data_y, predictions)

                dict_mse_RF[f'b{bootstrap}-w{warm_start}'].append(mse_rf)

                print(f"first {mse_rf}")

    if save_values:
        dict_mse_RF['range'] = list_n_estimators

        modelsaver.save_model(dict_mse_RF, "RF_graph_estimators_performance_dict")


# list_n_estimators = [i for i in range(10, 601, 10)]
# do_graph_estimators(list_n_estimators)


def plot_results():
    dict_rf = modelsaver.get_model("RF_graph_estimators_performance_dict_v2")

    list_n_estimators = dict_rf['range']
    # ------------------------------------------------------------------------------------
    # waarden = f'b{bootstrap}-w{warm_start}'
    names_bootstrap = ["Warm start", "No warm start"]

    fig_warm_boot, = plt.plot(list_n_estimators, dict_rf[f'b{True}-w{True}'])
    fig_notwarm_boot, = plt.plot(list_n_estimators, dict_rf[f'b{True}-w{False}'])

    plt.title("Performance Randomforest-Boostrap: number of estimators")
    plt.xlabel("Number of estimators")
    plt.ylabel("MSE")

    plt.legend([fig_warm_boot, fig_notwarm_boot], names_bootstrap)

    plt.show()
    # ----------------------------------------------------------------------------------------
    names_nobootstrap = ["Warm start", "No warm start"]

    fig_warm_nboot, = plt.plot(list_n_estimators, dict_rf[f'b{False}-w{True}'])
    fig_notwarm_nboot, = plt.plot(list_n_estimators, dict_rf[f'b{False}-w{False}'])

    plt.title("Performance Randomforest-No Bootstrap: number of estimators")
    plt.xlabel("Number of estimators")
    plt.ylabel("MSE")

    plt.legend([fig_warm_nboot, fig_notwarm_nboot], names_nobootstrap)

    plt.show()


def rf_only_percentage(save_model=False):
    n_estimators = 600
    warm_start = True

    # collecting data from files
    datamanager = prep.DataManager(column_fitting="opt_standard",
                                   model="VG")

    # the columns to retrieve from the file
    # list_columns = ["strike_price_percent", "interest_rate", "volatility", "maturity", "call/put"]
    list_columns = None

    X, y = datamanager.get_training_data(list_column_names=list_columns)
    test_data_X, test_data_y = datamanager.get_test_data(list_column_names=list_columns)

    scaler = preprocessing.StandardScaler().fit(X)

    X_scaled = scaler.transform(X)
    X_test_scaled = scaler.transform(test_data_X)

    rf_model = RandomForestRegressor(criterion="mse",
                                     max_features="auto",
                                     n_jobs=6,
                                     n_estimators=n_estimators,
                                     warm_start=warm_start,
                                     bootstrap=True,
                                     verbose=1,
                                     random_state=1179)

    rf_model.fit(X_scaled, y)

    # predictions = rf_model.predict(test_data_X)
    predictions = rf_model.predict(X_test_scaled)

    mse_rf = mean_squared_error(test_data_y, predictions)

    print(mse_rf)

    if save_model:
        modelsaver.save_model(rf_model, "RF_BS_only_percentages")


rf_only_percentage(save_model=False)
