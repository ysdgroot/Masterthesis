from keras.optimizers import adam, adadelta, adagrad, adam, adamax, nadam
from keras.activations import relu, tanh, sigmoid, elu, selu, softplus, softsign
from sklearn.model_selection import KFold
from sklearn import preprocessing
from Estimators import DataCollector as dc
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import modelsaver
from keras.layers import *
from keras.models import Sequential
import random
from Estimators import random_search_nn
import importlib.resources as pkg_resources
from sklearn.metrics import mean_squared_error

#

########################################################
# --------------- PARAMETERS --------------------------#
########################################################
activation_functions = ["relu", "softsign", "sigmoid", "elu"]
optimizer = "adam"  # works generally the best for all activation functions.


########################################################################################################################
# 1) Werken met een Multilayered Neural Network
# 2) Werken met 1 layer network

########################################################################################################################


def build_nn_model(input_size, size_layers, activations):
    """
    Builds and returns the neural network

    :param input_size:int, the input_size of the neural network
    :param size_layers: list, with each element the number of nodes of the layer
    :param activations: list, with each element the activation function taken for the layer
    :return:compiled neural network
    """
    nn_model = Sequential()
    for index, (size, activation) in enumerate(zip(size_layers, activations)):
        if index == 0:
            nn_model.add(Dense(units=size, activation=activation, input_shape=(input_size,)))
        else:
            nn_model.add(Dense(units=size, activation=activation))
        # nn_model.add(activation)

    nn_model.add(Dense(units=1, activation='linear'))

    nn_model.compile(optimizer="adam", loss='mean_squared_error')

    return nn_model


def cross_validation_nn(architecture, X, y,
                        cv=5,
                        batch_size=32,
                        epochs=50):
    """
    Cross validation of a Neural Network.

    :param architecture: dict, with keys "input_size", "size_layers" and "activations"
    :param X: DataFrame, the input
    :param y: DataFrame, the target output of the X values
    :param cv: int (default=5), number for the k-fold cross validation
    :param batch_size: int (default=32), batch size for the neural network
    :param epochs: int (default=50), epocht for the neural network
    :return: dict with keys "Train": list of the mse of the training samples
                            "Test": list of the mse of the test samples
                            "Mean": mean of the test samples
    """
    list_train_validation = []
    list_test_validation = []

    kfold = KFold(n_splits=cv)

    print(f"architecture: {architecture}")

    for index, (train, test) in enumerate(kfold.split(X)):
        if isinstance(X, pd.DataFrame):
            X_train, X_test, y_train, y_test = X.loc[train], X.loc[test], y.loc[train], y.loc[test]
        else:
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        input_size = architecture["input"]
        size_layers = architecture["size_layers"]
        activations = architecture["activations"]
        nn_model = build_nn_model(input_size, size_layers, activations)

        history_fit = nn_model.fit(X_train, y_train,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   verbose=2,
                                   validation_data=(X_test, y_test))

        list_train_validation.append(history_fit.history['loss'][-1])
        list_test_validation.append(history_fit.history['val_loss'][-1])

    return {"Train": list_train_validation,
            "Test": list_test_validation,
            "Mean": np.mean(list_test_validation)}


def cv_layers(n_random_samples,
              stock_model,
              option_type,
              cv=3,
              batch_size=100,
              epochs=50,
              random_state=4173,
              scale=True):
    """
    Cross validation of random neural networks

    :param n_random_samples: int, number of random neural networks
    :param stock_model: str, "BS", "VG" or "H"
    :param option_type: str, "opt_standard", "opt_asianmean", "opt_lookbackmin" or "opt_lookbackmax"
    :param cv: int (default=3), cross validations
    :param batch_size: int(default=100), batch size of the neural networks
    :param epochs: int(default=50), number of epochs for the neural networks
    :param random_state: int(default=4173)
    :param scale: bool(default=False), whenever to scale the data
    :return: list of dicts with keys
                            "n_layers": number of layers used,
                            "size_layers": list of size n_layers, with all the sizes,
                            "activations": list of size n_layers, with all the activation functions
                            "cv_result": dict with the Train and Test errors
    """
    activation_functions = ["relu", "softsign", "sigmoid", "elu"]

    datamanager = dc.DataManager(stockmodel=stock_model, option_type=option_type)
    X, y = datamanager.get_training_data()

    if scale:
        scaler = preprocessing.StandardScaler().fit(X, y)
        X = scaler.transform(X)

    results_fitting = []

    np.random.seed(random_state)

    for i in range(n_random_samples):
        first_layer_size = random.randrange(50, 301, 50)

        # 1, 2 or 3 layers
        n_hidden_layers = np.random.randint(1, 4)

        size_layers = [first_layer_size // ((i + 1) ** i) for i in range(n_hidden_layers)]

        activation_layers = random.choices(activation_functions, k=n_hidden_layers)

        architecture = {"size_layers": size_layers,
                        "activations": activation_layers,
                        "input": X.shape[1]}

        gen_error = cross_validation_nn(architecture, X, y, cv=cv, batch_size=batch_size, epochs=epochs)

        nn_model_values = {"n_layers": n_hidden_layers,
                           "size_layers": size_layers,
                           "activations": activation_layers,
                           "cv_result": gen_error}
        results_fitting.append(nn_model_values)

    return results_fitting


def main_cv():
    models = ["BS", "VG", "H"]
    columns_fitting = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    start_random_state = 594

    n_tests = 50
    scale = True

    # CV of the Black-Scholes formula first
    stockmodel = "BS"
    option = "opt_exact_standard"

    print(f"Start cv for {stockmodel}-{option}")
    random_state = start_random_state + 30
    results = cv_layers(n_tests, stockmodel, option, random_state=random_state, scale=scale, epochs=50, cv=3)

    string_scaled = '_scaled' if scale else ""
    modelsaver.save_model(results, f"NN-random_search_{stockmodel}_{option}{string_scaled}")

    for i, stockmodel in enumerate(models):
        for j, option in enumerate(columns_fitting):
            print(f"Start cv for {stockmodel}-{option}")
            random_state = start_random_state + 10 * i + j * 2
            results = cv_layers(n_tests, stockmodel, option, random_state=random_state, scale=scale, epochs=50, cv=3)

            string_scaled = '_scaled' if scale else ""
            modelsaver.save_model(results, f"NN-random_search_{stockmodel}_{option}{string_scaled}")
    print("End")


def train_real_data(symbol="SPX"):
    dm_real = dc.DataManagerRealData(symbol)
    X_train, y_train = dm_real.get_training_data()
    X_test, y_test = dm_real.get_test_data()

    # X_train, y_train, X_test, y_test = dm_real.get_random_training_test_data(n_samples=150000)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train, y_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # list_activations = [["softsign", "sigmoid"], ["softsign", "sigmoid", "relu"], ["elu", "relu"],["relu", "elu"], 2*["relu"]]
    # list_n_nodes = [[300, 150], [300, 150, 33], [300, 150],[300, 150],[300, 150]]

    list_activations = [2 * ['softplus'], 2 * ["softsign"], 2 * ['elu']]
    list_n_nodes = 3 * [[300, 150]]

    # # activations = ["softsign", "sigmoid"]
    # # activations = ["softsign", "sigmoid", "relu"]
    # activations = 3 * ['relu']
    # # activations = ["elu", "relu"]
    for activations, nodes in zip(list_activations, list_n_nodes):
        model = build_nn_model(X_train.shape[1], nodes, activations)

        history = model.fit(X_train, y_train,
                            batch_size=1000,
                            epochs=50,
                            verbose=2,
                            validation_data=(X_test, y_test))
        print(f"Activations {activations} -- Nodes {nodes}")
        print(history.history)


def main_real_data():
    for symbol in ["SPX", "SPXPM", "SX5E"]:
        print(symbol)
        train_real_data(symbol)


# ----------------------------------------------------------------------------------------------------------------------
def get_best_model(stockmodel, option_type):
    """
    Function to return the best NN model from the cross validations
    :param stockmodel: str, "BS", "VG" or "H"
    :param option_type: str, "opt_standard", "opt_asianmean", "opt_lookbackmin" or "opt_lookbackmax"
    :return: tuple of lists
            (size_layers, activations)
    """
    base_file_name = f"NN-random_search_{stockmodel}_{option_type}_scaled.p"

    full_file_name = pkg_resources.open_text(random_search_nn, base_file_name).name

    results = modelsaver.get_model(full_file_name)
    results.sort(key=lambda x: x["cv_result"]["Mean"])

    return results[0]['size_layers'], results[0]['activations']


def full_dataset(stockmodel, option_type, only_call=False, with_percentage=False, scale=True):
    """
    print the results of the performance over the full dataset for the given stock stockmodel and option type
    :param stockmodel: str, "BS", "VG" or "H"
    :param option_type: str, "opt_standard", "opt_asianmean", "opt_lookbackmin" or "opt_lookbackmax"
    :param only_call: bool (default=False), if the dataset only contains the call options
    :param with_percentage: bool (default=False),
            if the dataset needs to contain the percentage of the stock price and the strike
    :param scale: bool (default=False), if the dataset needs to be scaled
    """
    dm = dc.DataManager(stockmodel=stockmodel,
                        option_type=option_type,
                        only_call=only_call,
                        with_percent=with_percentage)
    X_train, y_train = dm.get_training_data()

    if scale:
        scaler = preprocessing.StandardScaler().fit(X_train, y_train)
        X_train = scaler.transform(X_train)

    size_layers, activations = get_best_model(stockmodel, option_type)

    nn_model = build_nn_model(X_train.shape[1], size_layers, activations)

    nn_model.fit(X_train, y_train, verbose=0,
                 batch_size=100,
                 epochs=50)

    X_test, y_test = dm.get_test_data()
    if scale:
        X_test = scaler.transform(X_test)

    y_pred = nn_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred=y_pred)

    print(f"MSE: {mse}")


def main_full_dataset(only_call=False, with_percentage=False, scale=True):
    models = ["BS", "VG", "H"]
    option_types = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    if only_call:
        print("Only for call options")

    model = "BS"
    option = "opt_exact_standard"
    full_dataset(model, option, only_call=only_call, with_percentage=with_percentage, scale=scale)

    for stockmodel in models:
        for option_type in option_types:
            print(f"{stockmodel}:{option_type}")
            full_dataset(stockmodel, option_type, only_call=only_call, with_percentage=with_percentage, scale=scale)


def part_dataset_like_gpr(stockmodel, option_type, only_call=False, with_percentage=False, scale=True):
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

    # get the training and test data
    dm = dc.DataManager(stockmodel=stockmodel,
                        option_type=option_type,
                        only_call=only_call,
                        with_percent=with_percentage)
    X_train, y_train, x_not_selected, y_not_selected = dm.get_random_training_data(n_samples=n_samples,
                                                                                   random_state=random_state,
                                                                                   get_not_selected_data=True)

    if scale:
        scaler = preprocessing.StandardScaler().fit(X_train, y_train)
        X_train = scaler.transform(X_train)

    size_layers, activations = get_best_model(stockmodel, option_type)

    nn_model = build_nn_model(X_train.shape[1], size_layers, activations)

    nn_model.fit(X_train, y_train, verbose=1,
                 batch_size=100,
                 epochs=100)

    X_test, y_test = dm.get_test_data()
    if scale:
        X_test = scaler.transform(X_test)
        x_not_selected = scaler.transform(x_not_selected)

    y_pred = nn_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred=y_pred)

    y_pred_not_selected = nn_model.predict(x_not_selected)
    mse_not_selected = mean_squared_error(y_not_selected, y_pred_not_selected)

    print(f"MSE(test data): {mse_test}")
    print(f"MSE(not selected): {mse_not_selected}")


def main_part_dataset_like_gpr(only_call=False, with_percentage=False, scale=True):
    """
    Prints for every option type from every stockmodel the mse of the best parameters of the cross validation

    :param only_call: bool (default=False), whenever to use only the call options
    """
    models = ["BS", "VG", "H"]
    option_types = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    for stockmodel in models:
        for option_type in option_types:
            print(f"{stockmodel}:{option_type}")
            part_dataset_like_gpr(stockmodel, option_type,
                                  only_call=only_call, with_percentage=with_percentage, scale=scale)


if __name__ == '__main__':
    print('Start')
    # main_cv()
    # main_full_dataset()
    # main_part_dataset_like_gpr()
    main_real_data()
