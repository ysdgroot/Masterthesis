from ModelsStock.BlackScholes import BlackScholes
from OptionModels.PlainVanilla import PlainVanilla
from OptionModels.EuropeanAsian import AsianMean
from OptionModels.EuropeanLookback import Lookback
from joblib import Parallel, delayed
import numpy as np
import csv

make_BS_data = True
make_VG_data = False
make_heston_data = False

n_datapoints = 50000

steps_per_maturity = 400
n_paths_optionpricing = 20000

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


########################################################################################################################
# ------------------------------- Black Scholes -----------------------------------------------------------------------#
########################################################################################################################
if make_BS_data:
    file_name = "Generated Data - BS model - 11_2_20.csv"
    seed_values = 42
    seed_paths = 73

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
                       "Seed paths": seed_paths}

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
    strike_prices = random_values["strike_price"]
    strike_prices_precentages = random_values["strike_price_percent"]


    # for parallelization
    def calculate_save_price(i):
        print("Datapoint {}".format(i))

        interest_rate = interest_rates[i]
        vol = volatilities[i]
        start_price = stock_prices[i]
        strike_price = strike_prices[i]
        strike_price_perc = strike_prices_precentages[i]
        maturity = maturities[i]

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
                                                      seed=seed_paths + i)

        # write datapoints in the csv-file
        values_rand = [start_price, strike_price, strike_price_perc, interest_rate, vol, maturity]

        values_call = values_rand + ['C'] + dict_option_values['C'] + [exact_value_call]
        values_put = values_rand + ['P'] + dict_option_values['P'] + [exact_value_put]

        write_to_file(file_name, values_call)
        write_to_file(file_name, values_put)


    # start collection datapoints in parallel (4 cores)
    Parallel(4)(delayed(calculate_save_price)(i) for i in range(n_datapoints))

########################################################################################################################
# ------------------------------- Variance Gamma ----------------------------------------------------------------------#
########################################################################################################################
if make_VG_data:
    file_name = "Generated Data - VG model.csv"
    seed_values = 42
    seed_paths = 73

    stock_price_bound = (90, 110)
    strike_price_bound = (0.4, 1.6)
    interest_rate_bound = (0.01, 0.035)
    maturity_bound = (1, 60)  # todo vragen wat ik hier best doe.
    volatility_bound = (0.01, 0.2)

    data_boundaries = {"Stock price": stock_price_bound,
                       "Strike price": strike_price_bound,
                       "Maturity": maturity_bound,
                       "Volatility": volatility_bound,
                       "Seed values": seed_values,
                       "Seed paths": seed_paths}

    column_names_values = ["stock_price", "strike_price", "strike_price_percent",
                           "interest_rate", "volatility", "maturity", "call/put"]

    col_names = column_names_values + column_names_options

    # adding last column for the theoretical value of the option, only for BS model
    col_names.append("opt_exact_standard")

    with open(file_name, 'w', newline='') as fd:
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
    strike_prices = random_values["strike_price"]
    strike_prices_precentages = random_values["strike_price_percent"]

    # set seed
    np.random.seed(seed=seed_paths)

    # todo: parallelliseer
    # start collection datapoints
    for i in range(n_datapoints):
        print("Datapoint {}".format(i))

        interest_rate = interest_rates[i]
        vol = interest_rates[i]
        start_price = stock_prices[i]
        strike_price = strike_prices[i]
        strike_price_perc = strike_prices_precentages[i]
        maturity = maturities[i]

        bs = BlackScholes(interest_rate, vol)

        # calculation exact option prices
        exact_value_call = bs.solution_call_option(start_price, strike_price, maturity, interest_rate, vol)
        exact_value_put = bs.solution_put_option(start_price, strike_price, maturity, interest_rate, vol)

        # start simulation and calculation of the different options
        dict_option_values = bs.get_price_simulations(options,
                                                      n_paths_optionpricing,
                                                      start_price,
                                                      maturity,
                                                      strike_price=strike_price,
                                                      option_type=['C', 'P'],
                                                      steps_per_maturity=steps_per_maturity)

        # write datapoints in the csv-file
        values = [start_price, strike_price, strike_price_perc, interest_rate, vol, maturity]

        values_call = values + ['C'] + dict_option_values['C'] + [exact_value_call]
        values_put = values + ['P'] + dict_option_values['P'] + [exact_value_put]

        write_to_file(file_name, values_call)
        write_to_file(file_name, values_put)
