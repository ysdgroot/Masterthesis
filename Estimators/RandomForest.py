from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from Estimators import preprocessing_data as prep
import matplotlib.pyplot as plt
import modelsaver
from sklearn import preprocessing
import csv
import numpy as np
from sklearn import tree

# from sklearn.tree import plot_tree
########################################################
# ---------------- PARAMETERS -------------------------#
########################################################
models = ["BS", "VG", "H"]
# models = ["BS"]
columns_fitting = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]
# columns_fitting = ["opt_lookbackmax"]

dict_column_to_option = {"opt_standard": "Standard",
                         "opt_asianmean": "Asian",
                         "opt_lookbackmin": "Lookback (min)",
                         "opt_lookbackmax": "Lookback (max)",
                         "opt_exact_standard": "Standard(theory)"}


########################################################################################################################

def plot_results(model, column_fitting, dict_results, list_estimators):
    # list_figures = []
    list_names_fig = []

    figure, ax = plt.subplots()

    for key, value in dict_results.items():
        fig, = ax.plot(list_estimators, value)
        # list_figures.append(fig)
        list_names_fig.append(key)

    ax.set_title(f"Performance {model}-{dict_column_to_option[column_fitting]} option")
    ax.set_ylabel("Mean squared error")
    ax.set_xlabel("Number of estimators")
    ax.legend(list_names_fig)

    figure.savefig(f"RF-{model}-{dict_column_to_option[column_fitting]}.png")


def rf_n_estimators(model="BS",
                    column_fitting="opt_exact_standard",
                    range_n_estimators=range(50, 1001, 50),
                    save_mse=True,
                    save_figure=True,
                    max_features="auto"):
    # todo: commentaar schrijven
    # models = ["BS", "VG", "H"]
    # columns_fitting = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    dict_option_types = {"opt_exact_standard": "SE",
                         "opt_standard": "S",
                         "opt_asianmean": "A",
                         "opt_lookbackmin": "Lmin",
                         "opt_lookbackmax": "Lmax"}

    list_results_train = []
    list_results_test = []
    list_oob_score = []

    datamanager = prep.DataManager(model=model,
                                   column_fitting=column_fitting)

    X, y = datamanager.get_training_data()
    X_test, y_test = datamanager.get_test_data()

    for n_estimator in range_n_estimators:
        rf_model = RandomForestRegressor(n_estimators=n_estimator,
                                         verbose=1,
                                         n_jobs=7,
                                         random_state=2458 + n_estimator,
                                         max_features=max_features,
                                         oob_score=True)
        rf_model.fit(X, y)

        mse_train = mean_squared_error(y, rf_model.predict(X))
        mse_test = mean_squared_error(y_test, rf_model.predict(X_test))
        oob_score = rf_model.oob_score_

        print(f'Train {mse_train}')
        print(f'Test {mse_test}')
        print(f'OOB score: {oob_score}')

        list_results_train.append(mse_train)
        list_results_test.append(mse_test)
        list_oob_score.append(oob_score)

    dict_result = {"Train": list_results_train,
                   "Test": list_results_test,
                   "oob_score": list_oob_score,
                   "n_estimators": range_n_estimators}

    if save_figure:
        plot_results(model,
                     column_fitting,
                     dict_results=dict_result,
                     list_estimators=range_n_estimators)
    if save_mse:
        modelsaver.save_model(dict_result, f"rf_50-1000-results_train_test-{model}-{dict_option_types[column_fitting]}"
                                           f"-{max_features}")


def test_scaling():
    datamanager = prep.DataManager(model="VG", column_fitting="opt_standard")

    X, y = datamanager.get_training_data()
    rf_normal = RandomForestRegressor(n_estimators=200, n_jobs=7, verbose=1)
    rf_normal.fit(X, y)

    print("Data not scaled")
    mse_score = mean_squared_error(y, rf_normal.predict(X))
    print(f"MSE score training data: {mse_score}")

    X_test, y_test = datamanager.get_test_data()

    mse_score_test = mean_squared_error(y_test, rf_normal.predict(X_test))
    print(f"mse score test data: {mse_score_test}")

    scaler = preprocessing.StandardScaler().fit(X, y)
    X = scaler.transform(X)
    rf_scaled = RandomForestRegressor(n_estimators=200, n_jobs=7, verbose=1)
    rf_scaled.fit(X, y)

    print("Only call options")
    print("Scaled data")
    mse_score = mean_squared_error(y, rf_scaled.predict(X))

    print(f"MSE score training data: {mse_score}")
    X_test = scaler.transform(X_test)

    mse_score_test = mean_squared_error(y_test, rf_scaled.predict(X_test))

    print(f"mse score test data: {mse_score_test}")


def full_rf_all_model_columns_n_estimators():
    # todo: commentaar schrijven
    max_features = ["auto", "log2", 5]
    for model in models:
        for column_fitting in columns_fitting:
            for max_feature in max_features:
                print(f"Model: {model} - {column_fitting} - {max_feature}")
                rf_n_estimators(model=model,
                                column_fitting=column_fitting,
                                range_n_estimators=range(50, 1001, 50),
                                save_mse=True,
                                save_figure=False,
                                max_features=max_feature)
                if model == "BS" and column_fitting == "opt_standard":
                    print(f"Model: BS - Exacte - {max_feature}")
                    rf_n_estimators(model="BS",
                                    column_fitting="opt_exact_standard",
                                    range_n_estimators=range(50, 1001, 50),
                                    save_mse=True,
                                    save_figure=False,
                                    max_features=max_feature)


def one_tree_visualisation():
    rf = RandomForestRegressor(n_estimators=100, max_features="auto", n_jobs=6, verbose=2)

    datamanger = prep.DataManager()

    X, y = datamanger.get_training_data()

    # Train
    rf.fit(X, y)
    # Extract single tree
    estimator = rf.estimators_[8]

    # tree.export_graphviz(estimator, out_file='tree.dot', feature_names=X.columns)
    # plot_tree(estimator, feature_names=X.columns)

    # , figsize=(4, 4), dpi=800
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    tree.plot_tree(rf.estimators_[8],
                   feature_names=X.columns,
                   max_depth=2,
                   filled=True)
    # plt.title("Random Forest: Decision Tree")
    fig.savefig('rf_individualtree.png')

    print(estimator.get_depth())


if __name__ == "__main__":
    # todo: deze test uitvoeren!!
    full_rf_all_model_columns_n_estimators()
    # test_scaling()
    # one_tree_visualisation()
