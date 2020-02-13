from ModelsStock.Heston import HestonModel
from OptionModels.PlainVanilla import PlainVanilla
import time
import csv
import math
import numpy as np
from joblib import Parallel, delayed

# Testing paths
# time_steps_per_maturities = [i for i in range(100, 1001, 100)]
# amount_paths = [i for i in range(1000, 20001, 1000)]

# TODO: controleer snelheid van Heston model, zodat alle waarden ook bepaald kunnen worden
time_steps_per_maturities = [i for i in range(100, 501, 100)]
amount_paths = [i for i in range(5000, 10001, 1000)]
write_comment_info_and_header = False

file_name = 'ParallelTest-steps and accuracy-Heston-v1.csv'
maturity = 10
interest_rate = 0.001
start_vol = 0.05
long_var = 0.05
rate_revert = 1.5
vol_of_vol = 0.6
correlation = -0.5

start_price = 100
strike_price = 100

number_iterations = 20

heston = HestonModel(interest_rate, start_vol, long_var, rate_revert, vol_of_vol, correlation)
option = PlainVanilla()

if write_comment_info_and_header:
    col_names = ['time_step', 'paths', 'time', 'option_value', 'variance']

    with open(file_name, 'w', newline='') as fd:
        fd.write("Variance Gamma model")
        fd.write('# Maturity = {} \n'.format(maturity))
        fd.write('# Interest_rate = {} \n'.format(interest_rate))
        fd.write("# Start_vol = {} \n".format(start_vol))
        fd.write("# long variance = {} \n".format(long_var))
        fd.write("# Rate mean reversion = {} \n".format(rate_revert))
        fd.write("# Volatility of vol = {} \n".format(vol_of_vol))
        fd.write("# Correlation = {} \n".format(correlation))
        fd.write("# Start_price = {} \n".format(start_price))
        fd.write("# Strike_price = {} \n".format(strike_price))
        fd.write("# Option = Plainvanilla \n")
        fd.write("# Number of iterations = {} \n".format(number_iterations))

        # write the header
        csv.writer(fd).writerow(col_names)


def function_per_amount_paths(amount):
    for time_step in time_steps_per_maturities:
        print("Amount {}, timestep = {} ".format(amount, time_step))

        for i in range(number_iterations):
            start = time.perf_counter()
            paths = heston.get_stock_prices(amount, start_price, maturity, time_step_per_maturity=time_step,
                                            seed=42 + i)
            end = time.perf_counter()
            # total_time += end - start
            total_time = end - start

            approx_call = option.get_price(paths, strike_price=strike_price)

            variance = np.var(paths[:, -1])

            temp_result = [time_step, amount, total_time, approx_call, variance]
            with open(file_name, 'a', newline='') as fd:
                writer = csv.writer(fd)
                writer.writerow(temp_result)


Parallel(n_jobs=4)(delayed(function_per_amount_paths)(amount) for amount in amount_paths)
