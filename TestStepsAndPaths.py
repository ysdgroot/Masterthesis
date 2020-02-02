from ModelsStock.BlackScholes import BlackScholes
from OptionModels.PlainVanilla import PlainVanilla
import time
import csv
import math
import numpy as np

# Testing paths
time_steps_per_maturities = [i for i in range(100, 1001, 100)]
amount_paths = [i for i in range(15000, 20001, 1000)]
add_header = False
write_comment_info = False

# time_steps_per_maturities = [10 + i * 10 for i in range(20)]
# amount_paths = [100 + i * 100 for i in range(15)]
file_name = 'Test-steps and accuracy-BS-v1.csv'
maturity = 10
interest_rate = 0.001
volatitlity = 0.1
start_price = 100
strike_price = 100

number_iterations = 20

BS = BlackScholes(interest_rate, volatitlity)
option = PlainVanilla()  # standard European Option

exact_call = BlackScholes.solution_call_option(start_price, strike_price, maturity, interest_rate, volatitlity)

# write some comments about the test
if write_comment_info:
    with open(file_name, 'w', newline='') as fd:
        fd.write('# Maturity = {} \n'.format(maturity))
        fd.write('# Interest_rate = {} \n'.format(interest_rate))
        fd.write("# Volatility = {} \n".format(volatitlity))
        fd.write("# Start_price = {} \n".format(start_price))
        fd.write("# Strike_price = {} \n".format(strike_price))
        fd.write("# Option = Plainvanilla \n")

# write header (if necessary)
if add_header:
    col_names = ['time_step', 'paths', 'time', 'accuracy_absolute', 'accuracy_normal', 'max_abs_diff', 'min_rel_diff',
                 'max_rel_diff', 'variance', 'maturity', 'interest_rate', 'volatility', 'exact_value']
    with open(file_name, 'a', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow(col_names)

# def get_variance_call(paths, strike_price):
#     return np.var(paths[:, -1] * (paths[:, -1] > strike_price))

for amount in amount_paths:
    for time_step in time_steps_per_maturities:
        print("Amount {}, timestep = {} ".format(amount, time_step))
        total_time = 0
        absolute_rel_diff = 0
        rel_diff = 0
        max_abs_diff = 0
        min_rel_diff = math.inf
        max_rel_diff = 0

        variance = 0
        for i in range(number_iterations):
            start = time.perf_counter()
            paths = BS.get_stock_prices(amount, start_price, maturity, time_step_per_maturity=time_step, seed=42+i)
            end = time.perf_counter()
            total_time += end - start

            approx_call = option.get_price(paths, strike_price=strike_price)
            absolute_rel_diff += abs(exact_call - approx_call) / exact_call
            rel_diff += (approx_call - exact_call) / exact_call

            max_abs_diff = max(max_abs_diff, absolute_rel_diff)
            min_rel_diff = min(min_rel_diff, rel_diff)
            max_rel_diff = max(max_rel_diff, rel_diff)

            #getting the variance of the 'call' function
            # variance += get_variance_call(paths, strike_price)

        # values for writing in csv file
        average_time = total_time / number_iterations
        average_rel_diff_abs = absolute_rel_diff / number_iterations
        average_rel_diff = rel_diff / number_iterations

        # avg_variance = variance / number_iterations

        temp_result = [time_step, amount, average_time, average_rel_diff_abs, average_rel_diff, max_abs_diff,
                       min_rel_diff, max_rel_diff,  maturity,
                       interest_rate, volatitlity, exact_call]
        with open(file_name, 'a', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(temp_result)
