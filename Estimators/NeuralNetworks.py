import keras
import value as value
from keras.models import Sequential
from keras.layers import Dense
import GeneratedData
import numpy as np
import pandas as pd
import ModelsStock.BlackScholes as BS
import math
from sklearn.metrics import mean_squared_error

########################################################
# ---------------- PARAMETERS --------------------------#
########################################################

n_layers = 1  # number of layers
n_nodes_layer = 5  # number of nodes per layer
activation_function = 'relu'  # TODO: bekijken welke de beste zou zijn voor tijdsafhankelijke data
optimizer = None  # TODO: bekijken welke de beste zou zijn voor tijdsafhankelijke data

# optimizers : "adam", 'sgd', "adamax"

# epochs? batch_size?
########################################################################################################################


# TODO: kurtosis testen om te werken met keras en hoe het precies werkt
# 1) Werken met een Multilayered Neural Network
# 2) Werken met 1 layer network
# 3) Cross-validation gebruiken voor het bepalen van hyperparameters (hoeveelheid neuronen en welke)

# Making the neural network (test-version)
model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=5))
# model.add(Dense(units=50, activation='relu'))
# model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adamax')

file_name_BS = "BS model.csv"
data_BS = pd.read_csv(file_name_BS, header=0, comment='#')

col_names_X = ["stock_price", "strike_price", "interest_rate", "volatility", "maturity", "call/put"]
col_names_X_new = ["stock_price", "strike_price", "interest_rate", "volatility", "maturity"]
#  "opt_asianmean" ,"opt_lookbackmin","opt_lookbackmax","opt_exact_standard", "option_standard"

col_name_y = ["opt_exact_standard"]

X = data_BS[col_names_X]

# change categorical data
X["call/put"] = X["call/put"].map({"C": 1, "P": -1})

X = X[X["call/put"] == 1]
X = X[col_names_X_new]

y = data_BS[col_name_y]
y = y[data_BS["call/put"] == "C"]

model.fit(X, y, epochs=30, batch_size=10)

# start fitting the NN to the data

# model.fit(X, y, epochs=5, batch_size=32)

# help(model)

loss_and_metrics = model.evaluate(X, y)
print(loss_and_metrics)

test_waarde = [100, 100, 0.001, 0.1, 10]
df = {"stock_price": [100], "strike_price": [100], "interest_rate": [0.01], "volatility": [0.1], "maturity": [10]}
# df = dict(zip(col_names_X, test_waarde))

prediction = pd.DataFrame(df)
print(prediction)

print(model.predict(prediction))

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
