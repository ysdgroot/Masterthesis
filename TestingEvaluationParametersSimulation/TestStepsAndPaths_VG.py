from ModelsStock.VarianceGamma import VarianceGamma
from OptionModels.PlainVanilla import PlainVanilla
from OptionModels.EuropeanAsian import AsianMean
from OptionModels.EuropeanLookback import Lookback
import time
import csv
import numpy as np
from joblib import Parallel, delayed

# Testing paths
time_steps_per_maturities = [i for i in range(100, 1001, 100)]
amount_paths = [i for i in range(1000, 20001, 1000)]
# test paths
# time_steps_per_maturities = [i for i in range(100, 201, 100)]
# amount_paths = [i for i in range(1000, 2001, 1000)]

write_header_to_files = [True, True, True]

# if the tests needs to be done, in order 'Standard, Asian, Lookback'
do_tests = [False, False, False]
# do_tests = [True, True, True]

number_iterations = 50

# The different file_name to write through
file_name = 'Datafiles/Test-steps and accuracy-VG-v1.csv'
file_name2 = 'Datafiles/Test-steps and accuracy-VG-v2-Asian.csv'
file_name3 = 'Datafiles/Test-steps and accuracy-VG-v3-Lookback.csv'

file_names = [file_name, file_name2, file_name3]

# The different file_name to write through
maturity = 10
interest_rate = 0.001
sigma = 0.25
nu = 0.75
theta = -0.2
start_price = 100
strike_price = 100

# Construct object of Variance Gamma method
VG = VarianceGamma(interest_rate, theta, sigma, nu)

# The different options types
option = PlainVanilla()
option2 = AsianMean()
option3 = Lookback()

options = [option, option2, option3]
option_names = ["Plainvanilla", "Asian", "Lookback"]


########################################################################################################################

def write_comment_info_and_header(file_n, option_name):
    col_names = ['time_step', 'paths', 'time', 'option_price', 'variance']

    with open(file_n, 'w', newline='') as fd:
        fd.write("Variance Gamma model \n")
        fd.write('# Maturity = {} \n'.format(maturity))
        fd.write('# Interest_rate = {} \n'.format(interest_rate))
        fd.write("# Theta = {} \n".format(theta))
        fd.write("# Nu = {} \n".format(nu))
        fd.write("# Sigma = {} \n".format(sigma))
        fd.write("# Start_price = {} \n".format(start_price))
        fd.write("# Strike_price = {} \n".format(strike_price))
        fd.write("# Option = {} \n".format(option_name))
        fd.write("# Number of iterations = {} \n".format(number_iterations))

        # write the header
        csv.writer(fd).writerow(col_names)


for bool_header, bool_test, file_n, option_n in zip(write_header_to_files, do_tests, file_names, option_names):
    if bool_header and bool_test:
        write_comment_info_and_header(file_n, option_n)


def func_per_time_step(time_step, amount_paths):
    print("Amount {}, timestep = {} ".format(amount_paths, time_step))

    for i in range(number_iterations):
        start = time.perf_counter()
        paths = VG.get_stock_prices(amount_paths, start_price, maturity, time_step_per_maturity=time_step, seed=42 + i)
        end = time.perf_counter()
        # total_time += end - start
        total_time = end - start

        for boolean_test, name_file, opt in zip(do_tests, file_names, options):
            if boolean_test:
                approx_call = opt.get_price(paths, maturity, interest_rate, strike_price=strike_price)

                variance = np.var(paths[:, -1])

                temp_result = [time_step, amount_paths, total_time, approx_call, variance]
                with open(name_file, 'a', newline='') as fd:
                    csv.writer(fd).writerow(temp_result)


def iteration_func(amount):
    for time_step in time_steps_per_maturities:
        func_per_time_step(time_step, amount)


Parallel(n_jobs=4)(delayed(iteration_func)(amount) for amount in amount_paths)
