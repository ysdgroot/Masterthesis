import numpy as np
import stockmodels


def get_paths_generated(n_paths: int,
                        model: stockmodels.StockModel,
                        maturity: int,
                        steps_per_maturity: int = 200,
                        seed: int = None):
    return model.get_stock_prices(n_paths,
                                  maturity=maturity,
                                  steps_per_maturity=steps_per_maturity,
                                  seed=seed)


def generated_different_paths_bs(n_different_values: int,
                                 n_paths_per_value: int,
                                 bounds_interest_rate: tuple,
                                 bounds_volatility: tuple,
                                 maturity: int,
                                 steps_per_maturity: int = 200,
                                 seed: int = None):
    if seed is not None:
        np.random.seed(seed)

    interest_rates = np.random.uniform(bounds_interest_rate[0], bounds_interest_rate[1], size=n_different_values)
    volatilities = np.random.uniform(bounds_volatility[0], bounds_volatility[1], size=n_different_values)

    all_paths = []
    for interest_rate, volatility in zip(interest_rates, volatilities):
        bs = stockmodels.BlackScholes(interest_rate, volatility)
        all_paths.extend(bs.get_stock_prices(n_paths_per_value,
                                             start_price=1,
                                             maturity=maturity,
                                             steps_per_maturity=steps_per_maturity).tolist())

    return np.array(all_paths), interest_rates, volatilities


n_different_values = 100
n_paths = 5

paths, interest_rates, volatilities = generated_different_paths_bs(n_different_values, n_paths, (0, 0.02), (0.1, 0.3),
                                                                   1)

# print(paths)

paths = np.array([[[element] for element in rij] for rij in paths])
print(paths.shape)

training_interest_rates = []
training_vol = []
for interest_rate, vol in zip(interest_rates, volatilities):
    training_interest_rates.append(np.tile(interest_rate, (n_paths, 1)))
    training_vol.append(np.tile(vol, (n_paths, 1)))

training_interest_rates = np.array(training_interest_rates).flatten()
training_vol = np.array(training_vol).flatten()

from keras.models import Sequential
from keras.layers import GRU, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

model_nn = Sequential()
# model_nn.add(GRU(2, batch_input_shape=(None, 201, 1)))
model_nn.add(LSTM(2, batch_input_shape=(None, 201, 1)))

model_nn.compile(loss='mean_squared_error', optimizer='adam')

target = np.vstack((training_vol, training_interest_rates)).transpose()

x_train, x_test, y_train, y_test = train_test_split(paths, target, test_size=0.2)

history = model_nn.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

predictions = model_nn.predict(x_test)
print(f'MSE1: {mean_squared_error(y_test[:, 0], predictions[:, 0])}')
print(f'MSE2: {mean_squared_error(y_test[:, 1], predictions[:, 1])}')

plt.plot(history.history['loss'])
plt.show()
