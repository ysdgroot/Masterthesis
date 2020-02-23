import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import ModelsStock.BlackScholes as BS
import math
from sklearn.metrics import mean_squared_error

########################################################
#---------------- PARAMETERS --------------------------#
########################################################

n_layers = 1    # number of layers
n_nodes_layer = 5   # number of nodes per layer
activation_function = 'relu'    # TODO: bekijken welke de beste zou zijn voor tijdsafhankelijke data
optimizer = None                # TODO: bekijken welke de beste zou zijn voor tijdsafhankelijke data

# epochs? batch_size?
########################################################################################################################



# TODO: nu testen om te werken met keras en hoe het precies werkt
# 1) Werken met een Multilayered Neural Network
# 2) Werken met 1 layer network
# 3) Cross-validation gebruiken voor het bepalen van hyperparameters (hoeveelheid neuronen en welke

# Making the neural network (test-version)
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=5))
model.add(Dense(units=1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

print(type(model))
print(type(model) == Sequential)

# start fitting the NN to the data

# model.fit(X, y, epochs=5, batch_size=32)

# help(model)

# loss_and_metrics = model.evaluate(X, y)
# print(loss_and_metrics)

#X_test = np.array([100, 100, 0.01, 0.1, 1])
# price = [100, 100]
# strike = [100, 102]
# interest = [0.001,0.001]
# vol = [0.1,0.1]
# mat = [1,1]
#
# data_dict = {'Price': price, 'Strike_price': strike, 'Interest_rate': interest,
#             'Volatility': vol, 'Maturity': mat}
# data = pd.DataFrame(data=data_dict)

# data_test = BS.get_random_data_and_solutions('C', 1000, [90, 110], [0.01, 0.03], [0.01, 0.2], [0.25, 5], [80, 120])
# X_test = data_test.drop('Value_option', axis=1)
# y_test = data_test['Value_option']
#
# pred_y = model.predict(X_test)

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
