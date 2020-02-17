from ModelsStock.GeneralStockModel import StockModel
import numpy as np
import itertools
from OptionModels.PlainVanilla import PlainVanilla


class HestonModel(StockModel):

    def __init__(self, interest_rate, start_volatility,  long_variance, rate_revert_to_long,
                 volatility_of_volatility, correlation_processes):
        """

        :param interest_rate:
        :param start_volatility:
        :param long_variance:
        :param rate_revert_to_long:
        :param volatility_of_volatility:
        :param correlation_processes: A number between [-1, 1] for the correlation between the brownian motions.
        """
        # todo: documentatie!!
        self.start_volatility = start_volatility
        self.interest_rate = interest_rate
        self.long_variance = long_variance
        self.rate_revert_to_long = rate_revert_to_long
        self.volatility_of_volatility = volatility_of_volatility

        if correlation_processes < -1 or correlation_processes > 1:
            raise ValueError("Incorrect correlation")
        self.correlation_processes = correlation_processes

    def get_stock_prices(self, amount_paths, start_price, maturity, time_step_per_maturity=100, seed=None):
        """
        Simulations of stock prices based on the Heston model.
        # todo: beschrijving van het heston model

        :param amount_paths: Positive integer.
                            This is the total number of paths generated.
        :param start_price: Positive float.
                            Is the starting price of the stock.
        :param maturity: Positive integer.
                        The total time period for the simulation.
                        The period of one payment of the interest_rate should be the same as maturity=1.
        :param time_step_per_maturity: A positive integer. (default = 100)
                                    The amount of small steps taken to represent 1 maturity passing.
                                    The higher the number te more accurate it represents the stock,
                                        but more time consuming
        :param seed: Positive integer. (default = 42)
                    For replication purposes, to get same 'random' values.
        :return: 2d numpy.array of all the generated paths and volatilities, based on the Heston model.
                shape:
                       (2, (amount, maturity * time_step_per_maturity + 1))
                First element is the generated paths, the second the generated volatilities of each path.
                Each row represents a different path, the columns the time.
                The first column is the start_price.


        """

        if seed is not None:
            np.random.seed(seed=seed)

        dt = 1 / time_step_per_maturity
        number_of_steps = maturity * time_step_per_maturity

        all_volatilities = []

        all_processes = self.get_weiner_processes_with_correlation(amount_paths,
                                                                   self.correlation_processes,
                                                                   maturity,
                                                                   time_step_per_maturity=time_step_per_maturity)

        weiner_stock_processes = all_processes[:, 0, :]
        weiner_volatility_processes = all_processes[:, 1, :]

        for j in range(amount_paths):
            volatilities = [self.start_volatility]
            weiner_volatility = weiner_volatility_processes[j]
            for i in range(number_of_steps):
                last_vol = volatilities[-1]
                not_negative_vol = max(last_vol, 0)

                dnu = self.rate_revert_to_long * (self.long_variance - not_negative_vol) * dt + \
                      self.volatility_of_volatility * \
                      np.sqrt(not_negative_vol) * weiner_volatility[i]
                volatilities.append(last_vol + dnu)
            all_volatilities.append(volatilities)
        # change it to a numpy array.
        all_volatilities = np.array(all_volatilities)

        # add first column of 0, as starting point.
        first_column = np.zeros((amount_paths, 1))
        weiner_stock_processes = np.append(first_column, weiner_stock_processes, axis=1)

        # Because of the Euler scheme, it is possible to get negative values for Volatitlity.
        # This cause problems because the square root is taken for the volatility.
        useable_vol = np.max(all_volatilities, 0)
        rate = self.interest_rate * dt
        vol = np.sqrt(useable_vol) * weiner_stock_processes

        # taking the product of the previous elements til the end.
        # not take first element of 'vol' because this is already 0, otherwise it wil get one time to mucht interest
        total_process = np.cumprod(1 + rate + vol[:, 1:], 1) * start_price

        first_column_stock = np.ones((amount_paths, 1)) * start_price
        total_process = np.append(first_column_stock, total_process, axis=1)
        print(total_process)

        # for j in range(amount_paths):
        #     stock_prices = [start_price]
        #     volatilities = [self.start_volatility]
        #
        #     for i in range(number_of_steps):
        #         last_price = stock_prices[-1]
        #         last_vol = volatilities[-1]
        #         not_negative_vol = max(last_vol, 0)
        #
        #         dS = last_price * (self.interest_rate * dt + np.sqrt(not_negative_vol) * weiner_stock[i])
        #         dnu = self.rate_revert_to_long * (self.long_variance - not_negative_vol) * dt + \
        #               self.volatility_of_volatility * \
        #               np.sqrt(not_negative_vol) * weiner_volatility[i]
        #
        #         # adding the next stock prices and volatilities
        #         stock_prices.append(stock_prices[-1] + dS)
        #         volatilities.append(volatilities[-1] + dnu)
        #     all_stock_prices.append(stock_prices)
        #     all_volatilities.append(volatilities)

        # return np.array(all_stock_prices)
        return total_process, all_volatilities

    @staticmethod
    def get_weiner_processes_with_correlation(n_paths, correlation, maturity, time_step_per_maturity=100):
        """
        Gives 2 Brownian motions which are correlated with each other.

        :param n_paths: A positive integer.
                        For the number of paths that needs to be generated.
                        Changes the output of this function.
        :param correlation: Value between -1 and 1.
                            This value is the correlation of the 2 brownian motions.
        :param maturity: Positive integer.
                        The total time period for the simulation.
                        The period of one payment of the interest_rate should be the same as maturity=1.
        :param time_step_per_maturity: A positive integer. (default = 100)
                                    The amount of small steps taken to represent 1 maturity passing.
                                    The higher the number te more accurate it represents the stock,
                                        but more time consuming
        :return: 2d numpy.array with shape:
                    (n_paths, 2, maturity * time_step_per_maturity)
                The rows are the brownian motions and together have correlation of the given value 'correlation'
        """

        dt = 1 / time_step_per_maturity
        number_of_steps = maturity * time_step_per_maturity

        # brownian_motions = np.random.randn(2, number_of_steps)
        brownian_motions = np.random.randn(n_paths, 2, number_of_steps)

        # declare the matrix to get the correlation
        matrix_pro = np.array([correlation, np.sqrt(1 - correlation ** 2)])

        # do dot-product between matrixes to get the correlated process of the first element of the 'brownian_motions'
        correlated_process = np.array(list(map(lambda x: matrix_pro.dot(x), brownian_motions)))

        # put everything in the correct form of matrix and the correct values(jumps) of the process.
        total_process_paths = np.array([[brownian_motions[i, 0], correlated_process[i]] for i in range(n_paths)]) \
                              * np.sqrt(dt)

        return total_process_paths

    @staticmethod
    def generate_random_variables(amount,
                                  stock_price_bound,
                                  strike_price_bound,
                                  maturity_bound,
                                  interest_rate_bound,
                                  start_vol_bound,
                                  long_variance_bound,
                                  rate_revert_to_long_bound,
                                  vol_of_vol_bound,
                                  correlation_bound,
                                  seed=None):

        """
        The generation of random variables for the Heston Model.

        :param amount: A positive integer.
                        The amount of different values, of each parameter.
        :param stock_price_bound:
        :param strike_price_bound:
        :param maturity_bound:
        :param interest_rate_bound:
        :param start_vol_bound:
        :param long_variance_bound:
        :param rate_revert_to_long_bound:
        :param vol_of_vol_bound:
        :param correlation_bound:
        :param seed:
        :return:
        """
        # todo: schrijven van documentatie

        def conversion_and_check(value):
            # convert value into a tuple in increasing order.
            # control if the values are positive
            if len(value) == 1:
                bounds = (value, value)
            elif len(value) >= 2:
                lower = min(value[0], value[1])
                upper = max(value[0], value[1])
                bounds = (lower, upper)
            else:
                raise TypeError

            # only 1 check is necessary, because this is the minimum.
            if bounds[0] < 0:
                raise ValueError
            return bounds

        # set seed if different from None.
        if seed is not None:
            np.random.seed(seed=seed)

        # conversion to a tuple, in increasing order and controls if the values are positive.
        stock_price_bound = conversion_and_check(stock_price_bound)
        strike_price_bound = conversion_and_check(strike_price_bound)
        maturity_bound = conversion_and_check(maturity_bound)
        interest_rate_bound = conversion_and_check(interest_rate_bound)
        start_vol_bound = conversion_and_check(start_vol_bound)
        long_variance_bound = conversion_and_check(long_variance_bound)
        rate_revert_to_long_bound = conversion_and_check(rate_revert_to_long_bound)
        vol_of_vol_bound = conversion_and_check(vol_of_vol_bound)
        correlation_bound = conversion_and_check(correlation_bound)

        # Check if the maturity is an integer.
        if type(maturity_bound[0]) is not int or type(maturity_bound[1]) is not int:
            raise ValueError

        # Check if correlation is between -1 and 1
        if correlation_bound[0] < -1 or correlation_bound[1]>1:
            raise ValueError("Values of correlation must be between -1 and 1")

        # random Integer selection for the maturities
        maturities = np.random.randint(maturity_bound[0], maturity_bound[1], amount)

        # random data selection, Uniform
        stock_prices = np.random.uniform(stock_price_bound[0], stock_price_bound[1], amount)
        strike_prices_percentage = np.random.uniform(strike_price_bound[0], strike_price_bound[1], amount)
        interest_rates = np.random.uniform(interest_rate_bound[0], interest_rate_bound[1], amount)
        start_vols = np.random.uniform(start_vol_bound[0], start_vol_bound[1], amount)
        long_variances = np.random.uniform(long_variance_bound[0], long_variance_bound[1], amount)
        rate_revert_to_longs = np.random.uniform(rate_revert_to_long_bound[0], rate_revert_to_long_bound[1], amount)
        vol_of_vols = np.random.uniform(vol_of_vol_bound[0], vol_of_vol_bound[1], amount)
        correlations = np.random.uniform(correlation_bound[0], correlation_bound[1], amount)

        # Take a percentage of the stock price
        strike_prices = stock_prices * strike_prices_percentage

        # making dictionary for each parameter
        data_dict = {"stock_price": stock_prices,
                     "strike_price": strike_prices,
                     "interest_rate": interest_rates,
                     "maturity": maturities,
                     "start_vol": start_vols,
                     "long_variance": long_variances,
                     "rate_revert_to_long": rate_revert_to_longs,
                     "vol_of_vol": vol_of_vols,
                     "correlation": correlations}

        return data_dict


import matplotlib.pyplot as plt

maturity = 1
interest_rate = 0.0319
start_vol = 0.010201
long_var = 0.019
rate_revert = 6.21
vol_of_vol = 0.61
correlation = -0.7

start_price = 100
strike_price = 100

# Construction object for the Heston model
heston = HestonModel(interest_rate, start_vol, long_var, rate_revert, vol_of_vol, correlation)

n_paths = 10000
model = heston.get_stock_prices(n_paths, 100, maturity, time_step_per_maturity=500)[0]
# plt.plot(model)
# plt.show()

option_standard = PlainVanilla()

print(option_standard.get_price(model, maturity, interest_rate))
