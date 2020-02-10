from ModelsStock.BlackScholes import BlackScholes
import ModelsStock.BlackScholes as BS
from ModelsStock.BlackScholes import BlackScholes
from OptionModels.PlainVanilla import PlainVanilla
from OptionModels.EuropeanAsian import AsianMean
from OptionModels.EuropeanLookback import Lookback
# from ModelsStock.VarianceGamma import VarianceGamma
# from ModelsStock.Heston import HestonModel
#
import numpy as np
from OptionModels.PlainVanilla import PlainVanilla
import timeit
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn.metrics import mean_squared_error

# ,,1.2484183531271091,,,57,C,276.50262638851757,184.71873542991236,301.4290227542398,0.0,66.95938980721127
# ,,1.581081399081321,,,29,C,92.87446790077998,75.02217462063409,152.125760199382,0.0,45.78125046985539

n_paths_optionpricing = 20000

start_price = 106.84148874797158
maturity = 29
strike_price = 168.92509050957412

interest_rate = 0.02598403392435606
vol = 0.156289493047819
# resultaat = 276.50262638851757
# theoretisch = 66.95938980721127


options = [PlainVanilla(False)]

bs = BlackScholes(interest_rate, vol)

for i in range(1):
    dict_option_values = bs.get_price_simulations(options,
                                                  n_paths_optionpricing,
                                                  start_price,
                                                  maturity,
                                                  interest_rate,
                                                  strike_price=strike_price,
                                                  option_type='C',
                                                  steps_per_maturity=100,
                                                  seed=None)

    print(dict_option_values)

print(BlackScholes.solution_call_option(start_price, strike_price, maturity, interest_rate, vol))

# data = BS.get_random_data_and_solutions('C', 10000, [80, 120], [0.01, 0.03], [0.01, 0.2], [0.25, 5], [70, 130])
# print('End new data')
#
# X = data.drop('Value_option', axis=1)
# y = data['Value_option']
#
# start = time.time()
# model_random_forest = RandomForestRegressor().fit(X, y)
# end = time.time()
# print('Time: ' + str(end - start))
#
# data_test = BS.get_random_data_and_solutions('C', 1000, [90, 110], [0.01, 0.03], [0.01, 0.2], [0.25, 5], [80, 120])
# X_test = data_test.drop('Value_option', axis=1)
# y_test = data_test['Value_option']
#
#
# pred_y = model_random_forest.predict(X_test)
#
#
# score = model_random_forest.score(X_test, y_test)
# print(score)
#
# mse = mean_squared_error(y_test, pred_y)
# print("Mean Squared Error:", mse)
#
# rmse = math.sqrt(mse)
# print("Root Mean Squared Error:", rmse)


######################################################################################################################
## Testing Stockmodel function call
#####################################################################################################################
# print("Testing correct function call")
# amount_paths = 1000
# start_price = 100
# maturity = 5
# strike_price = 100
#
# interest_rate = 0.001
# vol = 0.1
#
# bs = BlackScholes(interest_rate, vol)
# option = PlainVanilla()
#
# price = bs.get_price_simulations(option, amount_paths, start_price, maturity, strike_price=strike_price, option_type='Q')
# print("Price: {}".format(price))


######################################################################################################################
## Testing BlackScholes functions and Option function
#####################################################################################################################
# print("Testing BS functions")
#
# interest_rate = 0.001
# vol = 0.1
# start_price = 100
# strike_price = 100
# maturity = 10
# BS = BlackScholes(0.001, 0.1)
# option = PlainVanilla()
#
#
# exact_call = BlackScholes.solution_call_option(start_price, strike_price, maturity, interest_rate, vol)
# print("Exact value: %d " % exact_call)
#
# amount_paths = [100*i for i in range(1, 11)]
# time_steps_per_maturity = [100*i for i in range(1, 11)]
#
# for amount in amount_paths:
#     for time_step in time_steps_per_maturity:
#         paths = BS.get_stock_prices(amount, start_price, maturity, time_step_per_maturity=time_step)
#
#         approx_call = option.get_price(paths, strike_price=strike_price)
#         #absolute_rel_diff = abs(exact_call - approx_call) / exact_call
#
#         print("Amount: {} ; time_step: {}".format(amount, time_step))
#         #print("Absolute Relative diff: {} ".format(absolute_rel_diff))
#         print("Approximation: {} ".format(approx_call))
#
#
#
#

#######################################################################################################################
# Testing BlackScholes functions and Option function
######################################################################################################################
# print("Testing BS and Option functions")
#
# option = PlainVanilla()
#
# interest_rate = 0.001
# vol = 0.1
# start_price = 100
# strike_price = 100
# maturity = 5
#
# starting_prices = np.array([100 for i in range(6)])
# end_results_test = np.array([101, 98, 102, 103, 97, 96])
# path_stock = np.ones((6, 2))
# path_stock[:, 0] = starting_prices
# path_stock[:, 1] = end_results_test
#
#
# def test_oproep():
#     option.get_price_dict(path_stock, strike_price=strike_price,option_type='C')
#
#
# def test_direct():
#     option.get_price(path_stock, strike_price=strike_price,option_type='C')
#
#
# # print(option.get_price(path_stock, strike_price=strike_price, option_type='P'))
# tijd_oproep = timeit.timeit(test_oproep, number=10000)
# tijd_direct = timeit.timeit(test_direct, number=10000)
#
# print("Oproepen: {}".format(tijd_oproep))
# print("Direct: {}".format(tijd_direct))
#

#######################################################################################################################
# Testing shapes of return stocks
######################################################################################################################

# theta = 0.1
# nu = 0.1
# volatility = 0.1
# interest_rate = 0.001
# corr = 0.5
# # bs = BlackScholes(interest_rate, volatility)
# # vg = VarianceGamma(interest_rate, theta, volatility, nu)
# heston = HestonModel(interest_rate, volatility, volatility, theta, volatility, corr)
# 
# amount = 1000
# start_price = 100
# maturity = 7
# time_step_per_maturity = 200
# 
# #test1 = bs.get_stock_prices(amount, start_price, maturity, time_step_per_maturity=time_step_per_maturity)
# 
# # test2 = vg.get_stock_prices(amount, start_price, maturity, time_step_per_maturity=time_step_per_maturity)
# # test_process = vg.variance_process_brownian_motion(amount)
# 
# test3 = heston.get_stock_prices(amount, start_price, maturity, time_step_per_maturity=time_step_per_maturity)
# 
# print(test3)
# print("Shape")
# print(test3.shape)

######################################################################################################################
# TESTING: time consumption
######################################################################################################################
# import timeit
# import numpy as np
# import math
#
# x = np.random.uniform(1, 10000)
#
# def numpy_sqrt():
#     return np.sqrt(x)
#
# def numpy_emath_sqrt():
#     return np.emath.sqrt(x)
#
# def numpy_power():
#     return np.power(x, 2)
#
# def power():
#     return x ** 2
#
# n = 10000
#
# print(math.sqrt(x))
#
# t1 = timeit.timeit(numpy_power, number = n)
# print('Numpy power:', t1)
# t2 = timeit.timeit(power, number = n)
# print('Standard:', t2)
#
#
#
