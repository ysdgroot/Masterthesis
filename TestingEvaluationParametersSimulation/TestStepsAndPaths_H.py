from stockmodels import HestonModel
from options import PlainVanilla, AsianMean, Lookback
import time
import csv
import numpy as np
from multiprocessing import Manager, Pool

# Setting values for the amount of paths to generated and the step sizes (time_steps)
time_steps_per_maturities = [j for j in range(5, 100, 5)] + [i for i in range(100, 1001, 100)]
amount_paths = [i for i in range(1000, 20001, 1000)]

# if the tests needs to be done, in order 'Standard, Asian, Lookback'
write_header_to_files = [True, True, True]
do_tests = [True, True, True]

number_iterations = 50

# max number of paths generating during the process, to reduce RAM memory
max_paths_generating = 5000

# The different file_name to write through
file_name_standard = 'Datafiles/Test-steps and accuracy-H-Standard.csv'
file_name_asian = 'Datafiles/Test-steps and accuracy-H-Asian.csv'
file_name_lookback_min = 'Datafiles/Test-steps and accuracy-H-Lookback.csv'

file_names = [file_name_standard, file_name_asian, file_name_lookback_min]

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

# ----------------------------------------------------------------------------------------------------------------------

# Construction object for the Heston model
heston = HestonModel(interest_rate=interest_rate,
                     start_volatility=start_vol,
                     long_volatility=long_var,
                     rate_revert_to_long=rate_revert,
                     volatility_of_volatility=vol_of_vol,
                     correlation_processes=correlation)

# Different types of option_types
option_standard = PlainVanilla()
option_asian = AsianMean()
option_lookback_min = Lookback(lookback_min=True)

options = [option_standard, option_asian, option_lookback_min]
option_names = ["Plainvanilla", "Asian", "Lookback(min)", "Lockback(max)"]
dict_file_names = dict(zip(option_names, file_names))


########################################################################################################################


def partition_maker(total_number, value_splitter):
    """
    function to make partitions of size max 'value_splitter'.
    This is to use to split the number of paths generated at the same moment, to use less memory.

    :param total_number: positive int.
    :param value_splitter: positive int
    :return: list, with all the sizes of the partitions
    """
    deler, rest = divmod(total_number, value_splitter)

    values = [value_splitter] * deler
    if rest != 0:
        values += [rest]

    return values


def write_comment_info_and_header(file_n, option_name):
    """
    Directly write the information of the simulation as comment in the csv file

    :param file_n: str, full name of the file
    :param option_name: str, the name of the option
    :return: None
    """
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


def function_per_amount_paths(amount, queue):
    """
     Function to do the simulations in parallel and to write to the same file without mistakes.

     :param amount: positive int, number of paths that needs to be generated
     :param queue: Queue, for parallel writing
     :return: None
     """
    for time_step in time_steps_per_maturities:
        print(f"Amount {amount}, timestep = {time_step} ")

        for i in range(number_iterations):
            start = time.perf_counter()
            paths = heston.get_stock_prices(amount, start_price, maturity, steps_per_maturity=time_step, seed=42 + i)
            end = time.perf_counter()
            # total_time += end - start
            total_time = end - start

            for bool_test, file_name, option, opt_name in zip(do_tests, file_names, options, option_names):
                if bool_test:
                    approx_call, variance = option.get_price_option(paths, maturity, interest_rate,
                                                                    strike_price=strike_price)

                    temp_result = [time_step, amount, total_time, approx_call, variance]

                    queue.put((opt_name, temp_result))

    print(f"End {amount}")


def function_per_amount_paths_partitions(amount, queue):
    """
     Function to do the simulations in parallel and to write to the same file without mistakes.

     To use less memory, this function is written to generated less paths at once.

     :param amount: positive int, number of paths that needs to be generated
     :param queue: Queue, for parallel writing
     :return: None
     """
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
                total_time += end - start

                # do all the tests for each option if necessary
                for bool_test, file_name, option, opt_name in zip(do_tests, file_names, options, option_names):
                    if bool_test:
                        option_prices = dict_option_prices.get(opt_name, [])
                        option_prices.extend(option.get_prices_per_path(paths,
                                                                        maturity,
                                                                        interest_rate,
                                                                        strike_price=strike_price))
                        # put results in a temporary list
                        dict_option_prices[opt_name] = option_prices

            # writting to file when all the paths are generated and if necessary
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
        # get the first value, this will give the name of the option.
        # With the dictionary we get the file name
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
        job = pool.apply_async(function_per_amount_paths_partitions, (j, queue))
        jobs.append(job)

    for job in jobs:
        job.get()

    queue.put('kill')
    pool.close()
    pool.join()


if __name__ == "__main__":
    start = input("Start?(y/n)")
    if sum(do_tests) > 0 and start == 'y':
        # write header and comments if the simulation needs to be done
        for bool_header, bool_test, file_name, option_name in zip(write_header_to_files, do_tests, file_names,
                                                                  option_names):
            if bool_header and bool_test:
                write_comment_info_and_header(file_name, option_name)

        main_h()
