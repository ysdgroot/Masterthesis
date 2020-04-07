from stockmodels import VarianceGamma
from options import PlainVanilla, AsianMean, Lookback
import time
import csv
from multiprocessing import Manager, Pool

# Setting values for the amount of paths to generated and the step sizes (time_steps)
amount_paths = [i for i in range(1000, 20001, 1000)]
time_steps_per_maturities = [j for j in range(5, 100, 5)] + [i for i in range(100, 1001, 100)]

# if the tests needs to be done, in order 'Standard, Asian, Lookback'
write_header_to_files = [True, True, True]
do_tests = [True, True, True]

number_iterations = 50

# The different file_name to write through
file_name_standard = 'Datafiles/Test-steps and accuracy-VG-v1-1.csv'
file_name_asian = 'Datafiles/Test-steps and accuracy-VG-v2-1-Asian.csv'
file_name_lookback = 'Datafiles/Test-steps and accuracy-VG-v3-1-Lookback.csv'

file_names = [file_name_standard, file_name_asian, file_name_lookback]

# The different file_name to write through
maturity = 10
interest_rate = 0.001
volatility = 0.25
kurtosis = 0.75
skewness = -0.2
start_price = 100
strike_price = 100

# ----------------------------------------------------------------------------------------------------------------------

# Construct object of Variance Gamma method
VG = VarianceGamma(interest_rate=interest_rate,
                   volatility=volatility,
                   skewness=skewness,
                   kurtosis=kurtosis)

# The different option_types types
option_standard = PlainVanilla()
option_asian = AsianMean()
option_lookback = Lookback()

options = [option_standard, option_asian, option_lookback]
option_names = ["Plainvanilla", "Asian", "Lookback"]
dict_file_names = dict(zip(option_names, file_names))

########################################################################################################################


def write_comment_info_and_header(file_n, option_name):
    """
    Directly write the information of the simulation as comment in the csv file

    :param file_n: str, full name of the file
    :param option_name: str, the name of the option
    :return: None
    """
    col_names = ['time_step', 'paths', 'time', 'option_price', 'variance']

    with open(file_n, 'w', newline='') as fd:
        fd.write("# Variance Gamma model \n")
        fd.write(f'# Maturity = {maturity} \n')
        fd.write(f'# Interest_rate = {interest_rate} \n')
        fd.write(f"# Skewness = {skewness} \n")
        fd.write(f"# Kurtosis = {kurtosis} \n")
        fd.write(f"# Volatility = {volatility} \n")
        fd.write(f"# Start_price = {start_price} \n")
        fd.write(f"# Strike_price = {strike_price} \n")
        fd.write(f"# Option = {option_name} \n")
        fd.write(f"# Number of iterations = {number_iterations} \n")

        # write the header
        csv.writer(fd).writerow(col_names)


def func_per_time_step(amount, queue):
    """
     Function to do the simulations in parallel and to write to the same file without mistakes.

     :param amount: positive int, number of paths that needs to be generated
     :param queue: Queue, for parallel writing
     :return: None
     """
    for time_step in time_steps_per_maturities:
        print(f"Amount {amount}, timestep = {time_step}")
        for i in range(number_iterations):
            start = time.perf_counter()
            paths = VG.get_stock_prices(amount, start_price, maturity, steps_per_maturity=time_step, seed=42 + i)
            end = time.perf_counter()
            total_time = end - start

            for boolean_test, name_file, opt, opt_name in zip(do_tests, file_names, options, option_names):
                if boolean_test:
                    approx_call, variance = opt.get_price_option(paths, maturity, interest_rate,
                                                                 strike_price=strike_price)

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


def main_vg():
    manager = Manager()
    queue = manager.Queue()
    pool = Pool(5)

    # start file writer in other pool
    watcher = pool.apply_async(write_to_file_parallel, (queue,))
    jobs = []
    for j in amount_paths:
        job = pool.apply_async(func_per_time_step, (j, queue))
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

        main_vg()
