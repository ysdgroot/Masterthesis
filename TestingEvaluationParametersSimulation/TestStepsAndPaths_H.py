from ModelsStock.Heston import HestonModel
from OptionModels.PlainVanilla import PlainVanilla
from OptionModels.EuropeanAsian import AsianMean
from OptionModels.EuropeanLookback import Lookback
import time
import csv
import numpy as np
from multiprocessing import Manager, Pool

# Testing paths
# time_steps_per_maturities = [i for i in range(100, 1001, 100)]
# amount_paths = [i for i in range(1000, 20001, 1000)]

time_steps_per_maturities = [j for j in range(5, 100, 5)] + [i for i in range(100, 1001, 100)]
amount_paths = [i for i in range(1000, 20001, 1000)]

write_header_to_files = [True, True, True]
# write_header_to_files = [False, False, False]
# do_tests = [True, True, True]
do_tests = [False, False, False]

number_iterations = 50

# The different file_name to write through
file_name_standard = 'Datafiles/Test-steps and accuracy-H-v1-1.csv'
file_name_asian = 'Datafiles/Test-steps and accuracy-H-v2-1-Asian.csv'
file_name_lookback = 'Datafiles/Test-steps and accuracy-H-v3-1-Lookback.csv'

file_names = [file_name_standard, file_name_asian, file_name_lookback]

# The different file_name to write through
maturity = 10
interest_rate = 0.001
start_vol = 0.05
long_var = 0.05
rate_revert = 1.5
vol_of_vol = 0.6
correlation = -0.5

start_price = 100
strike_price = 100

# Construction object for the Heston model
heston = HestonModel(interest_rate, start_vol, long_var, rate_revert, vol_of_vol, correlation)

# Different types of options
option_standard = PlainVanilla()
option_asian = AsianMean()
option_lookback = Lookback()

options = [option_standard, option_asian, option_lookback]
option_names = ["Plainvanilla", "Asian", "Lookback"]
dict_file_names = dict(zip(option_names, file_names))

########################################################################################################################


def write_comment_info_and_header(file_n, option_name):
    col_names = ['time_step', 'paths', 'time', 'option_price', 'variance']

    with open(file_n, 'w', newline='') as fd:
        fd.write("# Heston model \n")
        fd.write('# Maturity = {} \n'.format(maturity))
        fd.write('# Interest_rate = {} \n'.format(interest_rate))
        fd.write("# Start_vol = {} \n".format(start_vol))
        fd.write("# long variance = {} \n".format(long_var))
        fd.write("# Rate mean reversion = {} \n".format(rate_revert))
        fd.write("# Volatility of vol = {} \n".format(vol_of_vol))
        fd.write("# Correlation = {} \n".format(correlation))
        fd.write("# Start_price = {} \n".format(start_price))
        fd.write("# Strike_price = {} \n".format(strike_price))
        fd.write("# Option = {} \n".format(option_name))
        fd.write("# Number of iterations = {} \n".format(number_iterations))

        # write the header
        csv.writer(fd).writerow(col_names)


# write header and comments if the work needs to be done
for bool_header, bool_test, file_name, option_name in zip(write_header_to_files, do_tests, file_names, option_names):
    if bool_header and bool_test:
        write_comment_info_and_header(file_name, option_name)


def function_per_amount_paths(amount, queue):
    for time_step in time_steps_per_maturities:
        print(f"Amount {amount}, timestep = {time_step} ")

        for i in range(number_iterations):
            start = time.perf_counter()
            paths = heston.get_stock_prices(amount, start_price, maturity, steps_per_maturity=time_step,
                                            seed=42 + i)
            end = time.perf_counter()
            # total_time += end - start
            total_time = end - start

            for bool_test, file_name, option, opt_name in zip(do_tests, file_names, options, option_names):
                if bool_test:
                    approx_call, variance = option.get_price(paths, maturity, interest_rate, strike_price=strike_price)

                    temp_result = [time_step, amount, total_time, approx_call, variance]

                    queue.put((opt_name, temp_result))


def write_to_file_parallel(queue):
    while 1:
        m = queue.get()
        if m == 'kill':
            break
        name_file = dict_file_names[m[0]]
        with open(name_file, 'a', newline='') as f:
            csv.writer(f).writerow(m[1])
            f.flush()


def main_h():
    manager = Manager()
    queue = manager.Queue()
    pool = Pool(5)

    # start file writer in other pool
    watcher = pool.apply_async(write_to_file_parallel, (queue,))
    jobs = []
    for j in amount_paths:
        job = pool.apply_async(function_per_amount_paths, (j, queue))
        jobs.append(job)

    for job in jobs:
        job.get()

    queue.put('kill')
    pool.close()
    pool.join()


if __name__ == "__main__":
    print('Start')
    main_h()
