from stockmodels import BlackScholes
from options import PlainVanilla, AsianMean, Lookback
import time
import csv
from multiprocessing import Manager, Pool

# Setting values for the amount of paths to generated and the step sizes (time_steps)
time_steps_per_maturities = [j for j in range(5, 100, 5)] + [i for i in range(100, 1001, 100)]
amount_paths = [i for i in range(1000, 20001, 1000)]

# if the tests needs to be done, in order 'Standard, Asian, Lookback'
write_header_to_files = [True, True, True]
do_tests = [True, True, True]

number_iterations = 50

# The different file_name to write through
file_name_standard = 'Datafiles/Black Scholes/Test-steps and accuracy-BS-Standard.csv'
file_name_asian = 'Datafiles/Black Scholes/Test-steps and accuracy-BS-Asian.csv'
file_name_lookback_min = 'Datafiles/Black Scholes/Test-steps and accuracy-BS-Lookback.csv'

file_names = [file_name_standard, file_name_asian, file_name_lookback_min]

maturity = 10
interest_rate = 0.001
volatitlity = 0.1
start_price = 100
strike_price = 100

# ----------------------------------------------------------------------------------------------------------------------

BS = BlackScholes(interest_rate, volatitlity)

# The different option_types types
option_standard = PlainVanilla()
option_asian = AsianMean()
option_lookback_min = Lookback(lookback_min=True)

options = [option_standard, option_asian, option_lookback_min]
option_names = ["Plainvanilla", "Asian", "Lookback(min)"]
dict_file_names = dict(zip(option_names, file_names))

is_plain_vanilla = [True, False, False]

# get price Black Scholes formula
exact_call = BlackScholes.solution_call_option(start_price, strike_price, maturity, interest_rate, volatitlity)


########################################################################################################################


def write_comment_info_and_header(file_n, option_name):
    """
    Directly write the information of the simulation as comment in the csv file

    :param file_n: str, full name of the file
    :param option_name: str, the name of the option
    :return: None
    """

    col_names = ['time_step', 'paths', 'time', 'option_price', 'variance']
    # different test for the BS stockmodel and plainvanilla option
    if option_name == "Plainvanilla":
        col_names += ["relative_diff", "absolute_diff", "exact_price"]

    with open(file_n, 'w', newline='') as fd:
        fd.write("# Black Scholes stockmodel \n")
        fd.write(f'# Maturity = {maturity} \n')
        fd.write(f'# Interest_rate = {interest_rate} \n')
        fd.write(f"# Volatitlity = {volatitlity}\n")
        fd.write(f"# Start_price = {start_price} \n")
        fd.write(f"# Strike_price = {strike_price} \n")
        fd.write(f"# Option = {option_name} \n")
        fd.write(f"# Number of iterations = {number_iterations} \n")

        # write the header
        csv.writer(fd).writerow(col_names)


def func_time_step(amount, queue):
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
            paths = BS.get_stock_prices(amount, start_price, maturity, steps_per_maturity=time_step, seed=42 + i)
            end = time.perf_counter()
            total_time = end - start

            # do all the tests for each option if necessary
            for boolean_test, name_file, opt, is_standard_opt, opt_name in zip(do_tests, file_names, options,
                                                                               is_plain_vanilla, option_names):
                if boolean_test:
                    approx_call, variance = opt.get_price_option(paths, maturity, interest_rate,
                                                                 strike_price=strike_price)

                    temp_result = [time_step, amount, total_time, approx_call, variance]

                    # When standard option, then calculate the relative difference
                    if is_standard_opt:
                        rel_diff = (approx_call - exact_call) / exact_call
                        rel_diff_abs = abs(rel_diff)

                        # adding the additional information in case if is a plainvanilla option.
                        # In this case we can compare the result with the theoretical price of the option.
                        temp_result.extend([rel_diff, rel_diff_abs, exact_call])

                    # Put results in queue a queue (later on this will be collected and written to a file
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


def main_bs():
    manager = Manager()
    queue = manager.Queue()
    pool = Pool(7)  # number of cores to use

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
    start = input("Start?(y/n)")
    if sum(do_tests) > 0 and start == 'y':
        # write header and comments if the simulation needs to be done
        for bool_header, bool_test, file_n, option_n in zip(write_header_to_files, do_tests, file_names, option_names):
            if bool_header and bool_test:
                write_comment_info_and_header(file_n, option_n)

        main_bs()
