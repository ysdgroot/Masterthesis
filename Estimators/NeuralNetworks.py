import keras
import value as value
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.losses as Loss
import GeneratedData
import numpy as np
import pandas as pd
import stockmodels
import math
from sklearn.metrics import mean_squared_error
import ModelSaver as MS
import importlib_resources as pkg_resources

########################################################
# --------------- PARAMETERS --------------------------#
########################################################

n_layers = 1  # number of layers
n_nodes_layer = 5  # number of nodes per layer
activation_function = 'relu'  # TODO: bekijken welke de beste zou zijn voor tijdsafhankelijke data
optimizer = None  # TODO: bekijken welke de beste zou zijn voor tijdsafhankelijke data

# optimizers : "adam", 'sgd', "adamax"

# epochs? batch_size?
########################################################################################################################


# 1) Werken met een Multilayered Neural Network
# 2) Werken met 1 layer network
# 3) Cross-validation gebruiken voor het bepalen van hyperparameters (hoeveelheid neuronen en welke)

model_not_saved = True

col_names_X = ["stock_price", "strike_price", "interest_rate", "volatility", "maturity", "call/put"]
col_names_X_new = ["stock_price", "strike_price", "interest_rate", "volatility", "maturity"]
#  "opt_asianmean" ,"opt_lookbackmin","opt_lookbackmax","opt_exact_standard", "option_standard"

col_name_y = ["opt_exact_standard"]

if model_not_saved:

    # Making the neural network (test-version)
    model = Sequential()
    model.add(Dense(units=150, activation='sigmoid', input_dim=6))
    model.add(Dense(units=50, activation='relu'))
    # model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    # file_name_BS = "BS model.csv"
    file_name_BS = "Generated Data - BS model -50k.csv"

    data_BS = pd.read_csv(file_name_BS, header=0, comment='#')

    X = data_BS[col_names_X]

    # change categorical data
    #    X["call/put"] = X["call/put"].map({"C": 1, "P": -1})
    data_training_callput = X["call/put"].map({"C": 1, "P": -1})

    # X = X[X["call/put"] == 1]
    # X = X[col_names_X_new]

    y = data_BS[col_name_y].to_numpy()

    data_array = np.array(
        [X["stock_price"].to_numpy(), X["strike_price"], X["interest_rate"], X["volatility"], X["maturity"],
         data_training_callput]).transpose()

    parts = [0, 10000, 20000, 30000, 40000]
    parts_2 = [10000, 20000, 30000, 40000, 50001]

    for start, end in zip(parts, parts_2):
        positions_fitting = range(start, end)
        X_fit = data_array[positions_fitting]
        y_fit = y[positions_fitting]
        model.fit(X_fit, y_fit, epochs=10, batch_size=10)

# start fitting the NN to the data

# model.fit(X, y, epochs=5, batch_size=32)

# help(model)

# loss_and_metrics = model.evaluate(data_array, y)
# print(loss_and_metrics)
else:
    model = MS.upload_model("NeuraalNetwerk2")

if model_not_saved:
    MS.save_model(model, "NeuraalNetwerk2")

test_data = pd.read_csv("Generated Data - BS model -Test data.csv", header=0, comment='#')

test_data_callput = test_data["call/put"].map({"C": 1, "P": -1})

y_test = test_data[col_name_y].to_numpy()

data_test_array = np.array(
    [test_data["stock_price"].to_numpy(), test_data["strike_price"], test_data["interest_rate"],
     test_data["volatility"], test_data["maturity"],
     test_data_callput]).transpose()

# print(test_waarde)
########################################################################################################################
print(model.evaluate(data_test_array, y_test))

test_waarde = np.array([[100], [100], [0.001], [0.1], [10], [1]]).transpose()
# df = {"stock_price": [100], "strike_price": [100], "interest_rate": [0.01], "volatility": [0.1], "maturity": [10]}
# df = dict(zip(col_names_X, test_waarde))
#
prediction = pd.DataFrame(test_waarde)
print(model.predict(test_waarde))

print(model.summary())

# score = model.evaluate(X_test, y_test) # niet nodig, want het gebruikt al de mse als evaluatie
# print(score)
#
# mse = mean_squared_error(y_test, pred_y)
# print("Mean Squared Error:", mse)
#
# rmse = math.sqrt(mse)
# print("Root Mean Squared Error:", rmse)
#
# print(model.summary())
