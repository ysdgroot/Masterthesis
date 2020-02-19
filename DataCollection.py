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
make_VG_data = True
make_heston_data = False

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
    return "# {} : {} \n".format(name, value)


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


def get_name_file(model, forward_pricing_bool):
    date_today = datetime.now().strftime('%d-%m-%Y')
    forward_bool = "(F)" if forward_pricing_bool else ""
    name_file = "Generated Data - {} model - {}{}.csv".format(model, date_today, forward_bool)
    return name_file


########################################################################################################################
# ------------------------------- Black Scholes -----------------------------------------------------------------------#
########################################################################################################################
if make_BS_data:
    forward_pricing = False
    model_name = "BS"

    file_name = get_name_file(model_name, forward_pricing)

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
                       "Interest_rate": interest_rate_bound,
                       "Volatility": volatility_bound,
                       "Seed values": seed_values,
                       "Seed paths": seed_paths,
                       "Forward pricing": forward_pricing}

    column_names_values = ["stock_price", "strike_price", "strike_price_percent",
                           "interest_rate", "volatility", "maturity", "call/put"]

    col_names = column_names_values + column_names_options

    # adding last column for the theoretical value of the option, only for BS model
    col_names.append("opt_exact_standard")

    with open(file_name, 'w', newline='') as fd:
        for key, val in dict_general_info.items():
            fd.write(get_comment_line(key, val))

        for key, val in data_boundaries.items():
            fd.write(get_comment_line(key, val))
        # writing the header
        csv.writer(fd).writerow(col_names)

    random_values = BlackScholes.generate_random_variables(n_datapoints,
                                                           stock_price_bound,
                                                           strike_price_bound,
                                                           maturity_bound,
                                                           interest_rate_bound,
                                                           volatility_bound,
                                                           seed=seed_values)

    # setting the values to a readable manner
    interest_rates = random_values["interest_rate"]
    volatilities = random_values["volatility"]
    maturities = random_values["maturity"]
    stock_prices = random_values["stock_price"]
    strike_prices_precentages = random_values["strike_price_percent"]

    # change the strike prices if we want to use the forward pricing
    strike_prices = random_values["strike_price"] if not forward_pricing \
        else stock_prices * np.exp(interest_rates * maturities) * strike_prices_precentages


    # for parallelization
    def calculate_save_price_bs(position, queue):
        print("BS Datapoint {}".format(position))

        interest_rate = interest_rates[position]
        vol = volatilities[position]
        start_price = stock_prices[position]
        strike_price = strike_prices[position]
        strike_price_perc = strike_prices_precentages[position]
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
    forward_pricing = True
    model_name = "VG"

    file_name = get_name_file(model_name, forward_pricing)
    # file_name = "Generated Data - VG model.csv"
    seed_values = 44
    seed_paths = 75

    stock_price_bound = (90, 110)
    strike_price_bound = (0.4, 1.6)
    interest_rate_bound = (0.01, 0.035)
    maturity_bound = (1, 60)
    volatility_bound = (0.01, 0.2)
    theta_bound = (-0.35, -0.05)
    sigma_bound = (0.05, 0.45)
    nu_bound = (0.55, 0.95)

    data_boundaries = {"Stock price": stock_price_bound,
                       "Strike price": strike_price_bound,
                       "Maturity": maturity_bound,
                       "Theta": volatility_bound,
                       "Sigma": sigma_bound,
                       "Nu": nu_bound,
                       "Seed values": seed_values,
                       "Seed paths": seed_paths,
                       "Forward pricing": forward_pricing}

    column_names_values = ["stock_price", "strike_price", "strike_price_percent",
                           "interest_rate", "theta", "sigma", "nu", "maturity", "call/put"]

    col_names = column_names_values + column_names_options

    with open(file_name, 'w', newline='') as fd:
        for key, val in data_boundaries.items():
            fd.write(get_comment_line(key, val))
        # writing the header
        csv.writer(fd).writerow(col_names)

    random_values = VarianceGamma.generate_random_variables(n_datapoints,
                                                            stock_price_bound,
                                                            strike_price_bound,
                                                            maturity_bound,
                                                            interest_rate_bound,
                                                            theta_bound,
                                                            sigma_bound,
                                                            nu_bound,
                                                            seed=seed_values)

    # setting the values to a readable manner
    interest_rates = random_values["interest_rate"]
    maturities = random_values["maturity"]
    stock_prices = random_values["stock_price"]
    strike_prices_precentages = random_values["strike_price_percent"]

    # change the strike prices if we want to use the forward pricing
    strike_prices = random_values["strike_price"] if not forward_pricing \
        else stock_prices * np.exp(interest_rates * maturities) * strike_prices_precentages

    thetas = random_values["theta"]
    sigmas = random_values["sigma"]
    nus = random_values["nu"]

    # set seed
    np.random.seed(seed=seed_paths)


    # for parallelization
    # start collection datapoints
    def calculate_save_price_vg(position, queue):
        print("VG Datapoint {}".format(position))

        interest_rate = interest_rates[position]
        theta = thetas[position]
        sigma = sigmas[position]
        nu = nus[position]
        start_price = stock_prices[position]
        maturity = maturities[position]
        strike_price = strike_prices[position]
        strike_price_perc = strike_prices_precentages[position]

        vg = VarianceGamma(interest_rate, theta, sigma, nu)

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
def main_bs():
    manager = Manager()
    queue = manager.Queue()
    pool = Pool(5)

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
    pool = Pool(5)

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


if __name__ == "__main__":
    print('Start')
    if make_BS_data:
        main_bs()
    if make_VG_data:
        main_vg()
