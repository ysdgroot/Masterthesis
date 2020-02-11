from ModelsStock.BlackScholes import BlackScholes
from OptionModels.PlainVanilla import PlainVanilla
import time
import csv
from joblib import Parallel, delayed

# Testing paths
time_steps_per_maturities = [i for i in range(100, 1001, 100)]
amount_paths = [i for i in range(1000, 20001, 1000)]
add_header = True
write_comment_info = True

file_name = 'ParallelTest-steps and accuracy-BS-v1.csv'
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
    # col_names = ['time_step', 'paths', 'time', 'accuracy_absolute', 'accuracy_normal', 'max_abs_diff', 'min_rel_diff',
    #              'max_rel_diff', 'maturity', 'interest_rate', 'volatility', 'exact_value']
    col_names = ['time_step', 'paths', 'time', 'accuracy_absolute', 'accuracy_normal', 'exact_value']
    with open(file_name, 'a', newline='') as fd:
        csv.writer(fd).writerow(col_names)


def func_time_step(time_step, amount):
    print("Amount {}, timestep = {} ".format(amount, time_step))

    for i in range(number_iterations):
        start = time.perf_counter()
        paths = BS.get_stock_prices(amount, start_price, maturity, steps_per_maturity=time_step, seed=42 + i)
        end = time.perf_counter()
        # total_time += end - start
        total_time = end - start

        approx_call = option.get_price(paths, maturity, interest_rate, strike_price=strike_price)
        # absolute_rel_diff += abs(exact_call - approx_call) / exact_call
        # rel_diff += (approx_call - exact_call) / exact_call
        rel_diff_abs = abs(exact_call - approx_call) / exact_call
        rel_diff = (approx_call - exact_call) / exact_call

        temp_result = [time_step, amount, total_time, rel_diff_abs, rel_diff, exact_call]
        with open(file_name, 'a', newline='') as fd:
            csv.writer(fd).writerow(temp_result)


def iteration_function(amount):
    for time_step in time_steps_per_maturities:
        func_time_step(time_step, amount)


# start of the parallelization of the test
Parallel(n_jobs=4)(delayed(iteration_function)(amount) for amount in amount_paths)