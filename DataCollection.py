from options import PlainVanilla, AsianMean, Lookback
from stockmodels import BlackScholes, VarianceGamma, HestonModel
from multiprocessing import Manager, Pool
import numpy as np
import csv
from datetime import datetime

# BS stockmodel
make_BS_data = False
forward_pricing_BS = False
test_data_bs = False

# VG stockmodel
make_VG_data = False
forward_pricing_VG = False
test_data_vg = False

# Heston stockmodel
make_heston_data = False
forward_pricing_heston = False
test_data_h = False

n_datapoints = 50000

steps_per_maturity = 200
n_paths_optionpricing = 15000

# partition the data_sizes, to use less RAM
max_path_generated = 5000

dict_general_info = {'n_datapoints (per type) ': n_datapoints,
                     'steps/maturity': steps_per_maturity,
                     'n_paths/option': n_paths_optionpricing}

option_types = [PlainVanilla(), AsianMean(), Lookback(lookback_min=True), Lookback(lookback_min=False)]
column_names_options = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]


def write_to_file_parallel(name_file, queue):
    """
    Write to a file while running in parallel.
    When simultaneously writing to the same file, some data can be lost.
    To reduce this loss, this function can be used during the parallelization to write in a csv-file
    :param name_file: str, with the file_names where to write (append values)
    :param queue: Manager.Queue() (package:  multiprocessing)
    :return: None
    """
    with open(name_file, 'a', newline='') as f:
        while 1:
            m = queue.get()
            if m == 'kill':
                break
            csv.writer(f).writerow(m)
            f.flush()


def create_name_file(model, forward_pricing_bool=False, testing=False):
    """
    Function to generate a file name, for a csv object, and will be stored in the "GeneratedData" folder.
    The name depends on stockmodel, the usage of the forward_pricing (percentage) and if the data is test/training data.
    :param model: str, normally "BS","VG" or "H".
                No problems will occur when using another modelname.
    :param forward_pricing_bool: bool, True: for the usage of the forward pricing; False: otherwise
    :param testing: bool, True: when the data is test data; False: data is training data.
    :return: str, the file name
            GeneratedData/Generated Data - {stockmodel} stockmodel - {date_today}{forward_bool}{test_file}.csv"

        date_today = '%d-%m-%y'
        forward_bool = (F) when 'forward_pricing_bool'=True
        test_file = "-Test data" when 'testing'=True

    """
    date_today = datetime.now().strftime('%d-%m-%Y')
    forward_bool = "(F)" if forward_pricing_bool else ""
    test_file = "-Test data" if testing else ""
    return f"GeneratedData/Generated Data - {model} stockmodel - {date_today}{forward_bool}{test_file}.csv"


def get_comment_line(name, value):
    """
    Getting string to write a comment (starting with #) in a csv file with a value, of the form:
        " # 'name' : 'value' "
    :param name: str, name of the value
    :param value: obj, value(s) for the name
    :return: str, of the form " # 'name' : 'value' "
    """
    return f"# {name} : {value} \n"


def write_comments(name_file, general_info, dict_data_boundaries, col_names):
    """
    Starting function to write the comments in the csv-file.

    :param name_file: str, file name where to write the comments
    :param general_info: dict, writing the general info of the simulation.
    :param dict_data_boundaries: dict, writing the boundaries for the random selection
    :param col_names: list(str), a list with the column names of the csv-file.
    :return: None
    """
    with open(name_file, 'w', newline='') as fd:
        for key, val in general_info.items():
            fd.write(get_comment_line(key, val))

        for key, val in dict_data_boundaries.items():
            fd.write(get_comment_line(key, val))
        # writing the header
        csv.writer(fd).writerow(col_names)


########################################################################################################################
# ------------------------------- Black Scholes -----------------------------------------------------------------------#
########################################################################################################################
if make_BS_data:

    model_name = "BS"

    file_name = create_name_file(model_name, forward_pricing_BS, testing=test_data_bs)

    seed_values = 3
    seed_paths = 6

    stock_price_bound = (90, 110) if not test_data_bs else (95, 105)
    strike_price_bound = (0.4, 1.6) if not test_data_bs else (0.6, 1.4)
    interest_rate_bound = (0, 0.035) if not test_data_bs else (0.001, 0.03)
    # maturity_bound = (1, 60) if not test_data else (1, 50)
    maturity_bound = (1, 25) if not test_data_bs else (1, 21)

    volatility_bound = (0.01, 0.45) if not test_data_bs else (0.015, 0.4)

    # change the seeds when using test_data, so there is no interference between test and training data
    if test_data_bs:
        seed_values += 2
        seed_paths += 2

    # for the comments in csv file
    data_boundaries = {"Stock price": stock_price_bound,
                       "Strike price": strike_price_bound,
                       "Maturity": maturity_bound,
                       "Interest rate": interest_rate_bound,
                       "Volatility": volatility_bound,
                       "Seed values": seed_values,
                       "Seed paths": seed_paths,
                       "Forward pricing": forward_pricing_BS}

    # header in the csv file
    column_names_values = ["stock_price",
                           "strike_price",
                           "strike_price_percent",
                           "interest_rate",
                           "volatility",
                           "maturity",
                           "call/put"]

    col_names = ["seed"] + column_names_values + column_names_options

    # adding last column for the theoretical value of the option, only for BS stockmodel
    col_names.append("opt_exact_standard")

    # write the info into the files
    write_comments(file_name, dict_general_info, data_boundaries, col_names=col_names)
    # ------------------------------------------------------------------------------------------------------------------

    # generating random values
    random_values = BlackScholes.generate_random_variables(n_datapoints,
                                                           stock_price_bound,
                                                           strike_price_bound,
                                                           maturity_bound,
                                                           interest_rate_bound,
                                                           volatility_bound,
                                                           forward_pricing=forward_pricing_BS,
                                                           seed=seed_values)

    # setting the values to a readable manner
    interest_rates = random_values["interest_rate"]
    volatilities = random_values["volatility"]
    maturities = random_values["maturity"]
    stock_prices = random_values["stock_price"]
    strike_prices_percentages = random_values["strike_price_percent"]

    # strike prices depends if usage of the forward pricing
    strike_prices = random_values["strike_price"]


    # for parallelization
    # start collection datapoints
    def calculate_save_price_bs(position, queue):
        print(f"BS Datapoint {position}")

        interest_rate = interest_rates[position]
        vol = volatilities[position]
        start_price = stock_prices[position]
        strike_price = strike_prices[position]
        strike_price_perc = strike_prices_percentages[position]
        maturity = maturities[position]

        bs = BlackScholes(interest_rate, vol)

        # calculation exact option prices
        exact_value_call = bs.solution_call_option(start_price, strike_price, maturity, interest_rate, vol)
        exact_value_put = bs.solution_put_option(start_price, strike_price, maturity, interest_rate, vol)

        # start simulation and calculation of the different option_types
        dict_option_values = bs.get_price_simulations(option_types,
                                                      n_paths_optionpricing,
                                                      start_price,
                                                      maturity,
                                                      interest_rate,
                                                      strike_price=strike_price,
                                                      option_type=['C', 'P'],
                                                      steps_per_maturity=steps_per_maturity,
                                                      seed=seed_paths + position,
                                                      max_path_generated=max_path_generated)

        # write datapoints in the csv-file
        values_rand = [seed_paths + position] + [start_price, strike_price, strike_price_perc, interest_rate, vol,
                                                 maturity]

        values_call = values_rand + ['C'] + dict_option_values['C'] + [exact_value_call]
        values_put = values_rand + ['P'] + dict_option_values['P'] + [exact_value_put]

        # putting in a queue so there will be no datapoints lost during the process of writing
        queue.put(values_call)
        queue.put(values_put)

########################################################################################################################
# ------------------------------- Variance Gamma ----------------------------------------------------------------------#
########################################################################################################################
if make_VG_data:
    model_name = "VG"

    file_name = create_name_file(model_name, forward_pricing_VG, testing=test_data_vg)
    seed_values = 53
    seed_paths = 84

    # add 1 to seed to get other values  when using forward pricing. Not strictly necessary.
    if forward_pricing_VG:
        seed_values += 1
        seed_paths += 1

    # Setting boundaries for each parameter of the Variance Gamma stockmodel.
    stock_price_bound = (90, 110) if not test_data_vg else (95, 105)
    strike_price_bound = (0.4, 1.6) if not test_data_vg else (0.6, 1.4)
    interest_rate_bound = (0, 0.035) if not test_data_vg else (0.001, 0.03)
    # maturity_bound = (1, 60) if not test_data_vg else (1, 50)
    maturity_bound = (1, 25) if not test_data_vg else (1, 21)

    skewness_bound = (-0.35, -0.05) if not test_data_vg else (-0.3, -0.1)
    volatility_bound = (0.05, 0.45) if not test_data_vg else (0.1, 0.4)
    kurtosis_bound = (0.55, 0.95) if not test_data_vg else (0.6, 0.9)

    # change the seeds when using test_data, so there is no interference between test and training data
    if test_data_vg:
        seed_values += 2
        seed_paths += 2

    # for the comments in csv file
    data_boundaries = {"Stock price": stock_price_bound,
                       "Strike price": strike_price_bound,
                       "Maturity": maturity_bound,
                       "Interest rate": interest_rate_bound,
                       "Volatility": volatility_bound,
                       "Skewness": skewness_bound,
                       "Kurtosis": kurtosis_bound,
                       "Seed values": seed_values,
                       "Seed paths": seed_paths,
                       "Forward pricing": forward_pricing_VG}

    # header in the csv file
    column_names_values = ["stock_price",
                           "strike_price",
                           "strike_price_percent",
                           "interest_rate",
                           "skewness",
                           "volatility",
                           "kurtosis",
                           "maturity",
                           "call/put"]

    col_names = ["seed"] + column_names_values + column_names_options

    # write the info into the files
    write_comments(file_name, dict_general_info, data_boundaries, col_names=col_names)
    # ------------------------------------------------------------------------------------------------------------------

    # generating random values
    random_values = VarianceGamma.generate_random_variables(n_datapoints,
                                                            stock_price_bound,
                                                            strike_price_bound,
                                                            maturity_bound,
                                                            interest_rate_bound,
                                                            skewness_bound,
                                                            volatility_bound,
                                                            kurtosis_bound,
                                                            forward_pricing=forward_pricing_VG,
                                                            seed=seed_values)

    # setting the values to a readable manner
    interest_rates = random_values["interest_rate"]
    maturities = random_values["maturity"]
    stock_prices = random_values["stock_price"]
    strike_prices_percentages = random_values["strike_price_percent"]

    # strike prices depends if usage of the forward pricing
    strike_prices = random_values["strike_price"]

    skewness = random_values["skewness"]
    volatilities = random_values["volatility"]
    kurtosis = random_values["kurtosis"]

    # for parallelization
    # start collection datapoints
    def calculate_save_price_vg(position, queue):
        print(f"VG Datapoint {position}")

        interest_rate = interest_rates[position]
        skewn = skewness[position]
        volatility = volatilities[position]
        kurtos = kurtosis[position]
        start_price = stock_prices[position]
        maturity = maturities[position]
        strike_price = strike_prices[position]
        strike_price_perc = strike_prices_percentages[position]

        vg = VarianceGamma(interest_rate, volatility, skewn, kurtos)

        # start simulation and calculation of the different option_types
        dict_option_values = vg.get_price_simulations(option_types,
                                                      n_paths_optionpricing,
                                                      start_price,
                                                      maturity,
                                                      interest_rate,
                                                      strike_price=strike_price,
                                                      option_type=['C', 'P'],
                                                      steps_per_maturity=steps_per_maturity,
                                                      seed=seed_paths + position,
                                                      max_path_generated=max_path_generated)

        # write datapoints in the csv-file
        values = [seed_paths + position] + \
                 [start_price, strike_price, strike_price_perc, interest_rate, skewn, volatility, kurtos, maturity]

        values_call = values + ['C'] + dict_option_values['C']
        values_put = values + ['P'] + dict_option_values['P']

        # put in Queue so no row will be lost when writing to it
        queue.put(values_call)
        queue.put(values_put)

########################################################################################################################
# ------------------------------- Heston Model ------------------------------------------------------------------------#
########################################################################################################################
if make_heston_data:
    model_name = "H"

    file_name = create_name_file(model_name, forward_pricing_heston, testing=test_data_h)
    seed_values = 45
    seed_paths = 76

    # add 1 to seed to get other values  when using forward pricing. Not strictly necessary.
    if forward_pricing_heston:
        seed_values += 1
        seed_paths += 1

    # set the boundaries for each parameter
    stock_price_bound = (90, 110) if not test_data_h else (95, 105)
    strike_price_bound = (0.4, 1.6) if not test_data_h else (0.6, 1.4)
    interest_rate_bound = (0, 0.035) if not test_data_h else (0.001, 0.03)
    # maturity_bound = (1, 60) if not test_data_h else (1, 50)
    maturity_bound = (1, 25) if not test_data_h else (1, 21)

    start_volatility_bound = (0.01, 0.1) if not test_data_h else (0.02, 0.09)
    long_variance_bound = (0.01, 0.1) if not test_data_h else (0.02, 0.09)
    rate_revert_to_long_bound = (1.4, 2.6) if not test_data_h else (1.5, 2.5)
    correlation_bound = (-0.85, -0.5) if not test_data_h else (-0.8, -0.55)
    vol_of_vol_bound = (0.45, 0.75) if not test_data_h else (0.5, 0.7)

    # change the seeds when using test_data, so there is no interference between test and training data
    if test_data_h:
        seed_values += 2
        seed_paths += 2

    # Setting values for commenting in files
    data_boundaries = {"Stock price": stock_price_bound,
                       "Strike price": strike_price_bound,
                       "Maturity": maturity_bound,
                       "Interest rate": interest_rate_bound,
                       "Start Volatility": start_volatility_bound,
                       "Long term volatility": long_variance_bound,
                       "Rate of mean reversion": rate_revert_to_long_bound,
                       "Volatility of volatility stock": vol_of_vol_bound,
                       "Correlation (brownian motions)": correlation_bound,
                       "Seed values": seed_values,
                       "Seed paths": seed_paths,
                       "Forward pricing": forward_pricing_heston}

    # setting column names for the csv-file
    column_names_values = ["stock_price",
                           "strike_price",
                           "strike_price_percent",
                           "interest_rate",
                           "maturity",
                           "start_vol",
                           "long_term_vol",
                           "rate_reversion",
                           "vol_of_vol",
                           "correlation",
                           "call/put"]

    col_names = ["seed"] + column_names_values + column_names_options

    # write the info into the files
    write_comments(file_name, dict_general_info, data_boundaries, col_names=col_names)
    # ------------------------------------------------------------------------------------------------------------------

    # generating random values
    random_values = HestonModel.generate_random_variables(n_datapoints,
                                                          stock_price_bound,
                                                          strike_price_bound,
                                                          maturity_bound,
                                                          interest_rate_bound,
                                                          start_volatility_bound,
                                                          long_variance_bound,
                                                          rate_revert_to_long_bound,
                                                          vol_of_vol_bound,
                                                          correlation_bound,
                                                          forward_pricing=forward_pricing_heston,
                                                          seed=seed_values)

    # setting the values to a readable manner
    interest_rates = random_values["interest_rate"]
    maturities = random_values["maturity"]
    stock_prices = random_values["stock_price"]
    strike_prices_percentages = random_values["strike_price_percent"]

    # strike prices depends if usage of the forward pricing
    strike_prices = random_values["strike_price"]

    # values specific of the Heston Model
    start_vols = random_values["start_vol"]
    long_variances = random_values["long_volatility"]
    rate_reversions = random_values["rate_revert_to_long"]
    vol_of_vols = random_values["vol_of_vol"]
    correlations = random_values["correlation"]

    # set seed
    np.random.seed(seed=seed_paths)

    # for parallelization
    # start collection datapoints
    def calculate_save_price_h(position, queue):
        print(f"Heston Datapoint {position}")

        interest_rate = interest_rates[position]
        start_price = stock_prices[position]
        maturity = maturities[position]
        strike_price = strike_prices[position]
        strike_price_perc = strike_prices_percentages[position]

        start_vol = start_vols[position]
        long_variance = long_variances[position]
        rate_reversion = rate_reversions[position]
        vol_of_vol = vol_of_vols[position]
        correlation = correlations[position]

        # making object of the Heston stockmodel
        heston = HestonModel(interest_rate, start_vol, long_variance, rate_reversion, vol_of_vol, correlation)

        # start simulation and calculation of the different option_types
        dict_option_values = heston.get_price_simulations(option_types,
                                                          n_paths_optionpricing,
                                                          start_price,
                                                          maturity,
                                                          interest_rate,
                                                          strike_price=strike_price,
                                                          option_type=['C', 'P'],
                                                          steps_per_maturity=steps_per_maturity,
                                                          seed=seed_paths + position,
                                                          max_path_generated=max_path_generated)

        # write datapoints in the csv-file
        values = [seed_paths + position] + \
                 [start_price, strike_price, strike_price_perc, interest_rate, maturity]

        values += [start_vol, long_variance, rate_reversion, vol_of_vol, correlation]

        values_call = values + ['C'] + dict_option_values['C']
        values_put = values + ['P'] + dict_option_values['P']

        # put in Queue so no row will be lost when writing to it
        queue.put(values_call)
        queue.put(values_put)

########################################################################################################################


def main_bs():
    manager = Manager()
    queue = manager.Queue()
    pool = Pool(7)

    # start file writer in other pool
    watcher = pool.apply_async(write_to_file_parallel, (file_name, queue))
    jobs = []
    for j in range(n_datapoints):
        job = pool.apply_async(calculate_save_price_bs, (j, queue))
        jobs.append(job)

    for job in jobs:
        job.get()

    queue.put('kill')
    pool.close()
    pool.join()


def main_vg():
    manager = Manager()
    queue = manager.Queue()
    pool = Pool(6)

    # start file writer in other pool
    watcher = pool.apply_async(write_to_file_parallel, (file_name, queue))
    jobs = []
    for j in range(n_datapoints):
        job = pool.apply_async(calculate_save_price_vg, (j, queue))
        jobs.append(job)

    for job in jobs:
        job.get()

    queue.put('kill')
    pool.close()
    pool.join()


def main_h():
    manager = Manager()
    queue = manager.Queue()
    pool = Pool(5)

    # start file writer in other pool
    watcher = pool.apply_async(write_to_file_parallel, (file_name, queue))
    jobs = []
    for j in range(n_datapoints):
        job = pool.apply_async(calculate_save_price_h, (j, queue))
        jobs.append(job)

    for job in jobs:
        job.get()

    queue.put('kill')
    pool.close()
    pool.join()


if __name__ == "__main__":
    print('Start')
    start = input("Sure to start?(y/n)")
    may_start = start == 'y'
    if make_BS_data and may_start:
        main_bs()
    if make_VG_data and may_start:
        main_vg()
    if make_heston_data and may_start:
        main_h()
