from stockmodels import BlackScholes, VarianceGamma, HestonModel
from options import PlainVanilla, AsianMean, Lookback
import time
import csv
import numpy as np
from multiprocessing import Manager, Pool

# Testing paths
# time_steps_per_maturities = [i for i in range(500, 1001, 100)]
# amount_paths = [i for i in range(1000, 20001, 1000)]

time_steps_per_maturities = [j for j in range(5, 100, 5)] + [i for i in range(100, 1001, 100)]
amount_paths = [i for i in range(1000, 20001, 1000)]

write_header_to_files = [True, True, True]
# write_header_to_files = [False, False, False]
# do_tests = [True, True, True]
do_tests = [False, False, False]

number_iterations = 50

# max number of paths generating during the process, to reduce RAM memory
max_paths_generating = 5000

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

# Different types of option_types
option_standard = PlainVanilla()
option_asian = AsianMean()
option_lookback = Lookback()

options = [option_standard, option_asian, option_lookback]
option_names = ["Plainvanilla", "Asian", "Lookback"]
dict_file_names = dict(zip(option_names, file_names))


########################################################################################################################


def partition_maker(total_number, value_splitter):
    deler, rest = divmod(total_number, value_splitter)

    values = [value_splitter] * deler
    if rest != 0:
        values += [rest]

    return values


def write_comment_info_and_header(file_n, option_name):
    col_names = ['time_step', 'paths', 'time', 'option_price', 'variance']

    with open(file_n, 'w', newline='') as fd:
        fd.write("# Heston model \n")
        fd.write(f'# Maturity = {maturity} \n')
        fd.write(f'# Interest_rate = {interest_rate} \n')
        fd.write(f"# Start_vol = {start_vol} \n")
        fd.write(f"# long variance = {long_var} \n")
        fd.write(f"# Rate mean reversion = {rate_revert} \n")
        fd.write(f"# Volatility of vol = {vol_of_vol} \n")
        fd.write(f"# Correlation = {correlation} \n")
        fd.write(f"# Start_price = {start_price} \n")
        fd.write(f"# Strike_price = {strike_price} \n")
        fd.write(f"# Option = {option_name} \n")
        fd.write(f"# Number of iterations = {number_iterations} \n")

        # write the header
        csv.writer(fd).writerow(col_names)


# write header and comments if the work needs to be done
for bool_header, bool_test, file_name, option_name in zip(write_header_to_files, do_tests, file_names, option_names):
    if bool_header and bool_test:
        write_comment_info_and_header(file_name, option_name)


# def function_per_amount_paths(amount, queue):
#     for time_step in time_steps_per_maturities:
#         print(f"Amount {amount}, timestep = {time_step} ")
#
#         for i in range(number_iterations):
#             start = time.perf_counter()
#             paths = heston.get_stock_prices(amount, start_price, maturity, steps_per_maturity=time_step, seed=42+i)
#             end = time.perf_counter()
#             # total_time += end - start
#             total_time = end - start
#
#             for bool_test, file_name, option, opt_name in zip(do_tests, file_names, option_types, option_names):
#                 if bool_test:
#                     approx_call, variance = option.get_price_option(paths, maturity, interest_rate,
#                                                                  strike_price=strike_price)
#
#                     temp_result = [time_step, amount, total_time, approx_call, variance]
#
#                     queue.put((opt_name, temp_result))
#
#     print(f"End {amount}")


def function_per_amount_paths(amount, queue):
    for time_step in time_steps_per_maturities:
        print(f"Amount {amount}, timestep = {time_step} ")

        n_paths_generating = partition_maker(amount, max_paths_generating)
        for i in range(number_iterations):
            total_time = 0

            dict_option_prices = dict()

            # set seed
            np.random.seed(42 + i)
            for n_paths in n_paths_generating:
                start = time.perf_counter()
                paths = heston.get_stock_prices(n_paths, start_price, maturity, steps_per_maturity=time_step)
                end = time.perf_counter()
                # total_time += end - start
                total_time += end - start

                for bool_test, file_name, option, opt_name in zip(do_tests, file_names, options, option_names):
                    if bool_test:
                        option_prices = dict_option_prices.get(opt_name, [])
                        option_prices.extend(option.get_prices_per_path(paths,
                                                                        maturity,
                                                                        interest_rate,
                                                                        strike_price=strike_price))
                        dict_option_prices[opt_name] = option_prices

            for bool_test, file_name, option, opt_name in zip(do_tests, file_names, options, option_names):
                if bool_test:
                    prices_option = dict_option_prices[opt_name]
                    approx_call = np.mean(prices_option)
                    variance = np.var(prices_option)

                    temp_result = [time_step, amount, total_time, approx_call, variance]

                    queue.put((opt_name, temp_result))
    print(f"End {amount}")


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
    start = input("Mag starten?(y/n)")
    if sum(do_tests) > 0 and start == 'y':
        main_h()
