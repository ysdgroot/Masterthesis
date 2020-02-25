from ModelsStock.BlackScholes import BlackScholes
from ModelsStock.VarianceGamma import VarianceGamma
from ModelsStock.Heston import HestonModel
from OptionModels.PlainVanilla import PlainVanilla
from OptionModels.EuropeanAsian import AsianMean
from OptionModels.EuropeanLookback import Lookback
from multiprocessing import Manager, Pool
import numpy as np
import csv
from datetime import datetime

make_BS_data = False
make_VG_data = False
make_heston_data = True

n_datapoints = 10000

steps_per_maturity = 200
n_paths_optionpricing = 15000

dict_general_info = {'n_datapoints (per type) ': n_datapoints,
                     'steps/maturity': steps_per_maturity,
                     'n_paths/option': n_paths_optionpricing}

options = [PlainVanilla(), AsianMean(), Lookback(lookback_min=True), Lookback(lookback_min=False)]
column_names_options = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]


def get_comment_line(name, value):
    """
    Getting string to write a comment (starting with #) in a csv file with a value, of the form:
        " # 'name' : 'value' "
    :param name: str, name of the value
    :param value: obj, value(s) for the name
    :return: str, of the form " # 'name' : 'value' "
    """
    return f"# {name} : {value} \n"


def write_to_file(filename, list_values):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list_values)


def write_to_file_parallel(name_file, queue):
    with open(name_file, 'a', newline='') as f:
        while 1:
            m = queue.get()
            if m == 'kill':
                break
            csv.writer(f).writerow(m)
            f.flush()


def create_name_file(model, forward_pricing_bool):
    date_today = datetime.now().strftime('%d-%m-%Y')
    forward_bool = "(F)" if forward_pricing_bool else ""
    return f"GeneratedData/Generated Data - {model} model - {date_today}{forward_bool}.csv"


def write_comments(name_file, general_info, dict_data_boundaries):
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
    forward_pricing_BS = False
    model_name = "BS"

    file_name = create_name_file(model_name, forward_pricing_BS)

    seed_values = 41
    seed_paths = 72

    stock_price_bound = (90, 110)
    strike_price_bound = (0.4, 1.6)
    interest_rate_bound = (0.01, 0.035)
    maturity_bound = (1, 60)
    volatility_bound = (0.01, 0.2)

    data_boundaries = {"Stock price": stock_price_bound,
                       "Strike price": strike_price_bound,
                       "Maturity": maturity_bound,
                       "Interest rate": interest_rate_bound,
                       "Volatility": volatility_bound,
                       "Seed values": seed_values,
                       "Seed paths": seed_paths,
                       "Forward pricing": forward_pricing_BS}

    column_names_values = ["stock_price", "strike_price", "strike_price_percent",
                           "interest_rate", "volatility", "maturity", "call/put"]

    col_names = column_names_values + column_names_options

    # adding last column for the theoretical value of the option, only for BS model
    col_names.append("opt_exact_standard")

    # write the info into the files
    write_comments(file_name, dict_general_info, data_boundaries)

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

        # start simulation and calculation of the different options
        dict_option_values = bs.get_price_simulations(options,
                                                      n_paths_optionpricing,
                                                      start_price,
                                                      maturity,
                                                      interest_rate,
                                                      strike_price=strike_price,
                                                      option_type=['C', 'P'],
                                                      steps_per_maturity=steps_per_maturity,
                                                      seed=seed_paths + position)

        # write datapoints in the csv-file
        values_rand = [start_price, strike_price, strike_price_perc, interest_rate, vol, maturity]

        values_call = values_rand + ['C'] + dict_option_values['C'] + [exact_value_call]
        values_put = values_rand + ['P'] + dict_option_values['P'] + [exact_value_put]

        # putting in a queue so there will be no datapoints lost during the process of writing
        queue.put(values_call)
        queue.put(values_put)

########################################################################################################################
# ------------------------------- Variance Gamma ----------------------------------------------------------------------#
########################################################################################################################
if make_VG_data:
    forward_pricing_VG = False
    model_name = "VG"

    file_name = create_name_file(model_name, forward_pricing_VG)
    seed_values = 43
    seed_paths = 74

    # add 1 to seed to get other values  when using forward pricing. Not strictly necessary.
    if forward_pricing_VG:
        seed_values += 1
        seed_paths += 1

    # Setting boundaries for each parameter of the Variance Gamma model.
    stock_price_bound = (90, 110)
    strike_price_bound = (0.4, 1.6)
    interest_rate_bound = (0.01, 0.035)
    maturity_bound = (1, 60)

    theta_bound = (-0.35, -0.05)
    sigma_bound = (0.05, 0.45)
    nu_bound = (0.55, 0.95)

    data_boundaries = {"Stock price": stock_price_bound,
                       "Strike price": strike_price_bound,
                       "Maturity": maturity_bound,
                       "Interest rate": interest_rate_bound,
                       "Theta": theta_bound,
                       "Sigma": sigma_bound,
                       "Nu": nu_bound,
                       "Seed values": seed_values,
                       "Seed paths": seed_paths,
                       "Forward pricing": forward_pricing_VG}

    column_names_values = ["stock_price", "strike_price", "strike_price_percent",
                           "interest_rate", "skewness", "volatility", "kurtosis", "maturity", "call/put"]

    col_names = column_names_values + column_names_options

    # write the info into the files
    write_comments(file_name, dict_general_info, data_boundaries)

    random_values = VarianceGamma.generate_random_variables(n_datapoints,
                                                            stock_price_bound,
                                                            strike_price_bound,
                                                            maturity_bound,
                                                            interest_rate_bound,
                                                            theta_bound,
                                                            sigma_bound,
                                                            nu_bound,
                                                            forward_pricing=forward_pricing_VG,
                                                            seed=seed_values)

    # setting the values to a readable manner
    interest_rates = random_values["interest_rate"]
    maturities = random_values["maturity"]
    stock_prices = random_values["stock_price"]
    strike_prices_percentages = random_values["strike_price_percent"]

    # strike prices depends if usage of the forward pricing
    strike_prices = random_values["strike_price"]

    thetas = random_values["skewness"]
    sigmas = random_values["volatility"]
    nus = random_values["kurtosis"]

    # set seed
    np.random.seed(seed=seed_paths)


    # for parallelization
    # start collection datapoints
    def calculate_save_price_vg(position, queue):
        print(f"VG Datapoint {position}")

        interest_rate = interest_rates[position]
        theta = thetas[position]
        sigma = sigmas[position]
        nu = nus[position]
        start_price = stock_prices[position]
        maturity = maturities[position]
        strike_price = strike_prices[position]
        strike_price_perc = strike_prices_percentages[position]

        vg = VarianceGamma(interest_rate, sigma, theta, nu)

        # start simulation and calculation of the different options
        dict_option_values = vg.get_price_simulations(options,
                                                      n_paths_optionpricing,
                                                      start_price,
                                                      maturity,
                                                      interest_rate,
                                                      strike_price=strike_price,
                                                      option_type=['C', 'P'],
                                                      steps_per_maturity=steps_per_maturity,
                                                      seed=seed_paths + position)

        # write datapoints in the csv-file
        values = [start_price, strike_price, strike_price_perc, interest_rate, theta, sigma, nu, maturity]

        values_call = values + ['C'] + dict_option_values['C']
        values_put = values + ['P'] + dict_option_values['P']

        # put in Queue so no row will be lost when writing to it
        queue.put(values_call)
        queue.put(values_put)

########################################################################################################################
# ------------------------------- Heston Model ------------------------------------------------------------------------#
########################################################################################################################
if make_heston_data:
    forward_pricing_heston = True
    model_name = "Heston"

    file_name = create_name_file(model_name, forward_pricing_heston)
    seed_values = 45
    seed_paths = 76

    # add 1 to seed to get other values  when using forward pricing. Not strictly necessary.
    if forward_pricing_heston:
        seed_values += 1
        seed_paths += 1

    # set the boundaries for each parameter
    stock_price_bound = (90, 110)
    strike_price_bound = (0.4, 1.6)
    interest_rate_bound = (0.01, 0.035)
    maturity_bound = (1, 60)

    start_volatility_bound = (0.01, 0.1)
    long_variance_bound = (0.01, 0.1)
    rate_revert_to_long_bound = (1.4, 2.6)
    correlation_bound = (-0.85, -0.5)
    vol_of_vol_bound = (0.45, 0.75)

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
    column_names_values = ["stock_price", "strike_price", "strike_price_percent",
                           "interest_rate", "maturity", "start_vol",
                           "long_term_vol", "rate_reversion", "vol_of_vol",
                           "correlation", "call/put"]

    col_names = column_names_values + column_names_options

    # write the info into the files
    write_comments(file_name, dict_general_info, data_boundaries)

    # creation of random values for the Heston Model
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

        # making object of the Heston model
        heston = HestonModel(interest_rate, start_vol, long_variance, rate_reversion, vol_of_vol, correlation)

        # start simulation and calculation of the different options
        dict_option_values = heston.get_price_simulations(options,
                                                          n_paths_optionpricing,
                                                          start_price,
                                                          maturity,
                                                          interest_rate,
                                                          strike_price=strike_price,
                                                          option_type=['C', 'P'],
                                                          steps_per_maturity=steps_per_maturity,
                                                          seed=seed_paths + position)

        # write datapoints in the csv-file
        values = [start_price, strike_price, strike_price_perc, interest_rate, maturity]

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
    pool = Pool(4)

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
    pool = Pool(4)

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
    pool = Pool(3)

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
    if make_BS_data:
        main_bs()
    if make_VG_data:
        main_vg()
    if make_heston_data:
        main_h()
