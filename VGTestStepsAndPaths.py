from ModelsStock.VarianceGamma import VarianceGamma
from OptionModels.PlainVanilla import PlainVanilla
import time
import csv
import math
import numpy as np
from joblib import Parallel, delayed

# Testing paths
time_steps_per_maturities = [i for i in range(100, 1001, 100)]
amount_paths = [i for i in range(1000, 20001, 1000)]
write_comment_info_and_header = True

file_name = 'ParallelTest-steps and accuracy-VG-v1.csv'
maturity = 10
interest_rate = 0.001
sigma = 0.25
nu = 0.75
theta = -0.2
start_price = 100
strike_price = 100

number_iterations = 20

VG = VarianceGamma(interest_rate, theta, sigma, nu)
option = PlainVanilla()

if write_comment_info_and_header:
    col_names = ['time_step', 'paths', 'time', 'option_value', 'variance']

    with open(file_name, 'w', newline='') as fd:
        fd.write("Variance Gamma model")
        fd.write('# Maturity = {} \n'.format(maturity))
        fd.write('# Interest_rate = {} \n'.format(interest_rate))
        fd.write("# Theta = {} \n".format(theta))
        fd.write("# Nu = {} \n".format(nu))
        fd.write("# Sigma = {} \n".format(sigma))
        fd.write("# Start_price = {} \n".format(start_price))
        fd.write("# Strike_price = {} \n".format(strike_price))
        fd.write("# Option = Plainvanilla \n")
        fd.write("Number of iterations = {}".format(number_iterations))

        # write the header
        csv.writer(fd).writerow(col_names)


def func_per_time_step(time_step):
    print("Amount {}, timestep = {} ".format(amount, time_step))

    for i in range(number_iterations):
        start = time.perf_counter()
        paths = VG.get_stock_prices(amount, start_price, maturity, time_step_per_maturity=time_step, seed=42 + i)
        end = time.perf_counter()
        # total_time += end - start
        total_time = end - start

        approx_call = option.get_price(paths, strike_price=strike_price)

        variance = np.var(paths[:, -1])

        temp_result = [time_step, amount, total_time, approx_call, variance]
        with open(file_name, 'a', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(temp_result)


for amount in amount_paths:
    Parallel(n_jobs=3)(delayed(func_per_time_step)(time_step) for time_step in time_steps_per_maturities)
