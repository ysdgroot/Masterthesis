from keras.optimizers import adadelta, adagrad, adam, adamax, nadam
from keras.activations import relu, tanh, sigmoid, elu, selu, softplus, softsign
from sklearn import preprocessing
from Estimators import preprocessing_data as prep
import matplotlib.pyplot as plt
import csv
import pandas as pd
import modelsaver
from keras.layers import *
from keras.models import Model, Sequential

########################################################
# --------------- PARAMETERS --------------------------#
########################################################
activation_functions = ["relu", "softsign", "sigmoid", "tanh"]
# activation_functions = ["softsign", "sigmoid"]
optimizer = "adam"
n_nodes = 200


########################################################################################################################


# 1) Werken met een Multilayered Neural Network
# 2) Werken met 1 layer network
# 3) Cross-validation gebruiken voor het bepalen van hyperparameters (hoeveelheid neuronen en welke)

########################################################################################################################


def plot_acc(history, optimizer, activation, nodes):
    figure, ax = plt.subplots()

    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title(f'Model loss-{optimizer}-{activation}-{nodes}')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper right')

    # Saving the figure that is made
    figure.savefig(f"Figures/Loss-{optimizer}-{activation}-{nodes}.png")
    # plt.show()


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
            model.add(Dense(units=n_node, input_dim=7))
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

            write_to_file([n_node, dict_optimizer_names[optimizer], dict_activation_names[activation], evaluation])


def do_optimizers_and_activations():
    datamanager = prep.DataManager(column_fitting="opt_exact_standard")
    X_train, y_train = datamanager.get_training_data()
    X_test, y_test = datamanager.get_test_data()

    file_name = "optimizers-activation-nodes-v3.csv"
    col_names = ["n_nodes", "optimizer", "activation", "mse"]

    with open(file_name, 'w', newline='') as fd:
        fd.write("# data = all 7 columns \n")
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
                                   column_fitting="opt_standard")
    # stock_price,strike_price,strike_price_percent,interest_rate,skewness,volatility,kurtosis,maturity,call/put
    # list_columns = ["strike_price_percent", "interest_rate", "volatility", "maturity", "call/put"]

    X_train, y_train = datamanager.get_training_data()
    X_test, y_test = datamanager.get_test_data()

    scaler = preprocessing.StandardScaler().fit(X_train, y_train)

    # scaling the values
    # X_train_scaled = scaler.transform(X_train)
    X_train_scaled = X_train
    # X_test_scaled = scaler.transform(X_test)
    X_test_scaled = X_test

    input_dimension = X_train_scaled.shape[1]

    for activation_func in activation_functions:
        model = Sequential()
        model.add(Dense(units=n_nodes,
                        activation=activation_func,
                        input_dim=input_dimension))
        # model.add(Dense(units=100,
        #                 activation='sigmoid'))
        # model.add(Dense(units=50,
        #                 activation='softsign'))
        # model.add(Dense(units=10,
        #                 activation='relu'))
        model.add(Dense(units=1, activation='linear'))

        model.compile(optimizer=optimizer, loss='mean_squared_error')

        history = model.fit(X_train_scaled, y_train,
                            batch_size=64,
                            epochs=100,
                            verbose=2,
                            validation_data=(X_test_scaled, y_test))

        figure, ax = plt.subplots()

        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])

        plt.show()


do_optimize_NN()
