from ModelsStock.BlackScholes import BlackScholes
from OptionModels.PlainVanilla import PlainVanilla
from OptionModels.EuropeanAsian import AsianMean
from OptionModels.EuropeanLookback import Lookback
import time
import csv
from multiprocessing import Manager, Pool

# Testing paths
time_steps_per_maturities = [j for j in range(5, 100, 5)] + [i for i in range(100, 1001, 100)]
# time_steps_per_maturities = [i for i in range(5, 100, 5)]
amount_paths = [i for i in range(1000, 20001, 1000)]

write_header_to_files = [True, True, True]
# if the tests needs to be done, in order 'Standard, Asian, Lookback'
# do_tests = [False, False, False]
do_tests = [True, True, True]

# The different file_name to write through
file_name_standard = 'Datafiles/Test-steps and accuracy-BS-v1-1.csv'
file_name_asian = 'Datafiles/Test-steps and accuracy-BS-v2-1-Asian.csv'
file_name_lookback = 'Datafiles/Test-steps and accuracy-BS-v3-1-Lookback.csv'

file_names = [file_name_standard, file_name_asian, file_name_lookback]

file_name = file_name_standard

maturity = 10
interest_rate = 0.001
volatitlity = 0.1
start_price = 100
strike_price = 100

number_iterations = 50

BS = BlackScholes(interest_rate, volatitlity)

# The different options types
option_standard = PlainVanilla()
option_asian = AsianMean()
option_lookback = Lookback()

options = [option_standard, option_asian, option_lookback]
option_names = ["Plainvanilla", "Asian", "Lookback"]
is_plain_vanilla = [True, False, False]
dict_file_names = dict(zip(option_names, file_names))

# get price Black Scholes formula
exact_call = BlackScholes.solution_call_option(start_price, strike_price, maturity, interest_rate, volatitlity)


########################################################################################################################


def write_comment_info_and_header(file_n, option_name, plain_vanilla_option=False):
    col_names = ['time_step', 'paths', 'time', 'option_price', 'variance']
    if plain_vanilla_option:
        col_names += ["relative_diff", "absolute_diff", "exact_price"]

    with open(file_n, 'w', newline='') as fd:
        fd.write("# Black Scholes model \n")
        fd.write(f'# Maturity = {maturity} \n')
        fd.write(f'# Interest_rate = {interest_rate} \n')
        fd.write(f"# Volatitlity = {volatitlity}")
        fd.write(f"# Start_price = {start_price} \n")
        fd.write(f"# Strike_price = {strike_price} \n")
        fd.write(f"# Option = {option_name} \n")
        fd.write(f"# Number of iterations = {number_iterations} \n")

        # write the header
        csv.writer(fd).writerow(col_names)


for bool_header, bool_test, file_n, option_n, is_standard in zip(write_header_to_files, do_tests, file_names,
                                                                 option_names, is_plain_vanilla):
    if bool_header and bool_test:
        write_comment_info_and_header(file_n, option_n, plain_vanilla_option=is_standard)


def func_time_step(amount, queue):
    for time_step in time_steps_per_maturities:
        print(f"Amount {amount}, timestep = {time_step} ")
        for i in range(number_iterations):
            start = time.perf_counter()
            paths = BS.get_stock_prices(amount, start_price, maturity, steps_per_maturity=time_step, seed=42 + i)
            end = time.perf_counter()
            total_time = end - start

            for boolean_test, name_file, opt, is_standard_opt, opt_name in zip(do_tests, file_names, options,
                                                                               is_plain_vanilla, option_names):
                if boolean_test:
                    approx_call, variance = opt.get_price(paths, maturity, interest_rate, strike_price=strike_price)

                    temp_result = [time_step, amount, total_time, approx_call, variance]

                    # When standard option, then caculate the relative difference
                    if is_standard_opt:
                        rel_diff = (approx_call - exact_call) / exact_call
                        rel_diff_abs = abs(rel_diff)

                        # adding the additional information in case if is a plainvanilla option.
                        # In this case we can compare the result with the theoretical price of the option.
                        temp_result.extend([rel_diff, rel_diff_abs, exact_call])

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


def main_bs():
    manager = Manager()
    queue = manager.Queue()
    pool = Pool(5)

    # start file writer in other pool
    watcher = pool.apply_async(write_to_file_parallel, (queue,))
    jobs = []
    for j in amount_paths:
        job = pool.apply_async(func_time_step, (j, queue))
        jobs.append(job)

    for job in jobs:
        job.get()

    queue.put('kill')
    pool.close()
    pool.join()


if __name__ == "__main__":
    print('Start')
    if sum(do_tests) > 0:
        main_bs()
