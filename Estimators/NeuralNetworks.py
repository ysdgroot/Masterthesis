from keras.optimizers import adadelta, adagrad, adam, adamax, nadam
from keras.activations import relu, tanh, sigmoid, elu, selu, softplus, softsign
from sklearn.model_selection import KFold, cross_val_score
from sklearn import preprocessing
from Estimators import preprocessing_data as prep
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import modelsaver
from keras.layers import *
from keras.models import Sequential
import random
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

########################################################
# --------------- PARAMETERS --------------------------#
########################################################
activation_functions = ["relu", "softsign", "sigmoid", "elu"]
optimizer = "adam"  # works generally the best for all activation functions.


########################################################################################################################
# 1) Werken met een Multilayered Neural Network
# 2) Werken met 1 layer network

########################################################################################################################


def plot_acc(history, optimizer, activation, nodes, save_figure=False):
    """
    # todo: commentaar schrijven
    :param history:
    :param optimizer:
    :param activation:
    :param nodes:
    :return:
    """

    figure, ax = plt.subplots()

    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title(f'Model loss-{optimizer}-{activation}-{nodes}')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper right')

    if save_figure:
        # Saving the figure that is made
        figure.savefig(f"Figures/Loss-{optimizer}-{activation}-{nodes}.png")
    else:
        plt.show()


def write_to_file(listvalues, name_file):
    # name_file = "optimizers-activation-nodes-v3.csv"
    with open(name_file, 'a', newline='') as f:
        csv.writer(f).writerow(listvalues)


def optimizers_and_activations(n_node, X_train, y_train, X_test, y_test, name_file):
    optimizers = [adadelta(), adagrad(), adam(), adamax(), nadam()]
    names_opt = ["adadelta", "adagrad", "adam", "adamax", "nadam"]
    dict_optimizer_names = dict(zip(optimizers, names_opt))

    activation_layers = [Activation(relu),
                         Activation(tanh),
                         Activation(sigmoid),
                         Activation(elu),
                         Activation(selu),
                         Activation(softplus),
                         Activation(softsign),
                         PReLU()]
    names_activation = ["ReLu", "tanh", "sigmoid", "elu", "selu", "softplus", "softsign", "PReLu"]
    dict_activation_names = dict(zip(activation_layers, names_activation))

    for optimizer in optimizers:
        for activation in activation_layers:
            print(f"nodes: {n_node}; optimizer: {dict_optimizer_names[optimizer]}; "
                  f"activation: {dict_activation_names[activation]}")

            model = Sequential()
            model.add(Dense(units=n_node, input_dim=X_train.shape[1]))
            model.add(activation)
            model.add(Dense(units=1, activation='linear'))

            model.compile(optimizer=optimizer, loss='mean_squared_error')

            history = model.fit(X_train, y_train,
                                epochs=100,
                                batch_size=64,
                                verbose=2,
                                validation_data=(X_test, y_test))

            plot_acc(history, dict_optimizer_names[optimizer], dict_activation_names[activation], n_node)

            evaluation = model.evaluate(X_test, y_test)

            write_to_file([n_node, dict_optimizer_names[optimizer], dict_activation_names[activation], evaluation],
                          name_file)


def do_optimizers_and_activations():
    datamanager = prep.DataManager(column_fitting="opt_exact_standard")
    X_train, y_train = datamanager.get_training_data()
    X_test, y_test = datamanager.get_test_data()

    file_name = "optimizers-activation-nodes-v3-newversion_input6.csv"
    col_names = ["n_nodes", "optimizer", "activation", "mse"]

    with open(file_name, 'w', newline='') as fd:
        fd.write("# data = 6 columns (without percentage), dataversion 2 (maturity <= 25) \n")
        fd.write("# fitting: opt_exact_standard \n")

        csv.writer(fd).writerow(col_names)

    n_nodes = [i for i in range(50, 201, 50)]

    for j in n_nodes:
        optimizers_and_activations(j,
                                   X_train=X_train,
                                   y_train=y_train,
                                   X_test=X_test,
                                   y_test=y_test,
                                   name_file=file_name)


def do_optimize_NN():
    # opt_exact_standard
    # "stock_price", "strike_price"

    models = ["BS", "VG", "H"]
    options = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    datamanager = prep.DataManager(model="H",
                                   column_fitting="opt_lookbackmax")
    # stock_price,strike_price,strike_price_percent,interest_rate,skewness,volatility,kurtosis,maturity,call/put
    # list_columns = ["strike_price_percent", "interest_rate", "volatility", "maturity", "call/put"]

    X_train, y_train = datamanager.get_training_data()
    X_test, y_test = datamanager.get_test_data()

    scaler = preprocessing.StandardScaler().fit(X_train, y_train)

    # scaling the values
    X_train_scaled = scaler.transform(X_train)
    # X_train_scaled = X_train
    X_test_scaled = scaler.transform(X_test)
    # X_test_scaled = X_test

    input_dimension = X_train_scaled.shape[1]

    for activation_func in activation_functions:
        print(activation_func)

        model = Sequential()
        model.add(Dense(units=200,
                        activation=activation_func,
                        input_dim=input_dimension))
        model.add(Dense(units=100,
                        activation=activation_func))
        model.add(Dense(units=50,
                        activation=activation_func))
        # model.add(Dense(units=10,
        #                 activation='relu'))
        model.add(Dense(units=1, activation='linear'))

        model.compile(optimizer=optimizer, loss='mean_squared_error')

        history = model.fit(X_train_scaled, y_train,
                            batch_size=32,
                            epochs=100,
                            verbose=2,
                            validation_data=(X_test_scaled, y_test))

        figure, ax = plt.subplots()

        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])

        plt.show()


def build_nn_model(input_size, size_layers, activations):
    """
    # todo commentaar
    :param input_size:
    :param size_layers:
    :param activations:
    :return:
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
                        epochs=100):
    list_train_validation = []
    list_test_validation = []

    kfold = KFold(n_splits=cv)

    # todo: controleer!
    for index, (train, test) in enumerate(kfold.split(X)):
        X_train, X_test, y_train, y_test = X.loc[train], X.loc[test], y.loc[train], y.loc[test]

        nn_model = build_nn_model(architecture)

        history_fit = nn_model.fit(X_train, y_train,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   verbose=2,
                                   validation_data=(X_test, y_test),
                                   workers=3,
                                   use_multiprocessing=True)

        list_train_validation.append(history_fit.history['loss'][-1])
        list_test_validation.append(history_fit.history['val_loss'][-1])

    return {"Train": list_train_validation,
            "Test": list_test_validation,
            "Mean": np.mean(list_test_validation)}


def cv_random_architecture(n_random_samples,
                           stock_model,
                           option_type,
                           cv=5,
                           batch_size=32,
                           epochs=100,
                           random_state=7883):
    activation_functions = ["relu", "softsign", "sigmoid", "elu"]
    # optimizer = "adam"

    datamanager = prep.DataManager(model=stock_model, column_fitting=option_type)
    X, y = datamanager.get_training_data()

    results_fitting = []

    np.random.seed(random_state)

    for i in range(n_random_samples):
        n_hidden_layers = np.random.randint(1, 5)
        size_layers = [random.randrange(10, 200, 10) for i in range(n_hidden_layers)]

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


def test_multiregression():
    datamanager = prep.DataManager("BS")

    columns_x = ["stock_price", "strike_price", "interest_rate", "volatility", "maturity", "call/put"]
    columns_y = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    df = datamanager.get_full_dataframe(test_data=False, change_call_put=True)

    X_train = df[columns_x]
    y_train = df[columns_y]

    scaler = preprocessing.StandardScaler().fit(X_train, y_train)
    X_train = scaler.transform(X_train)

    df_test = datamanager.get_full_dataframe(test_data=True, change_call_put=True)

    X_test = df_test[columns_x]
    y_test = df_test[columns_y]

    X_test = scaler.transform(X_test)

    size_layers = [200, 100, 50, 10]
    activations = ["softsign", "softsign", "softsign", "softsign"]
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    nn_model = Sequential()
    for index, (size, activation) in enumerate(zip(size_layers, activations)):
        if index == 0:
            nn_model.add(Dense(units=size, activation=activation, input_shape=(input_size,)))
        else:
            nn_model.add(Dense(units=size, activation=activation))
        # nn_model.add(activation)

    nn_model.add(Dense(units=output_size, activation='linear'))

    nn_model.compile(optimizer="adam", loss='mean_squared_error')

    history = nn_model.fit(X_train, y_train,
                           batch_size=100,
                           epochs=100,
                           verbose=2,
                           validation_data=(X_test, y_test))

    plot_acc(history=history, optimizer="adam", activation="softsign", nodes=size_layers)

    print(nn_model.predict(X_test))


# todo: bekijken van multi-regressie, dus alle optie types tegelijk trainen
if __name__ == '__main__':
    print('Start')
    # do_optimizers_and_activations()
    # do_optimize_NN()
    # resultaten = cv_random_architecture(2, "BS", "opt_standard", batch_size=64, epochs=100)
    # print(resultaten)
    test_multiregression()
