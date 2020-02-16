from ModelsStock.GeneralStockModel import StockModel
import numpy as np


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
        :return: 2d numpy.array of all the generated paths, based on the Black Scholes model.
                shape:
                        (amount, maturity * time_step_per_maturity + 1)
                Each row represents a different path, the columns the time.
                The first column is the start_price.
        """

        if seed is not None:
            np.random.seed(seed=seed)

        dt = 1 / time_step_per_maturity
        number_of_steps = maturity * time_step_per_maturity

        all_stock_prices = []
        all_volatilities = []

        for j in range(amount_paths):
            stock_prices = [start_price]
            volatilities = [self.start_volatility]

            weiner_stock, weiner_volatility = self.get_weiner_processes_with_correlation(self.correlation_processes,
                                                                                         maturity,
                                                                                         time_step_per_maturity=time_step_per_maturity)

            for i in range(number_of_steps):
                last_price = stock_prices[-1]
                last_vol = volatilities[-1]
                not_negative_vol = max(last_vol, 0)

                dS = last_price * (self.interest_rate * dt + np.sqrt(last_vol) * weiner_stock[i])
                dnu = self.rate_revert_to_long * (self.long_variance - not_negative_vol) * dt + \
                      self.volatility_of_volatility * \
                      np.sqrt(not_negative_vol) * weiner_volatility[i]

                # adding the next stock prices and volatilities
                stock_prices.append(stock_prices[-1] + dS)
                volatilities.append(volatilities[-1] + dnu)
            all_stock_prices.append(stock_prices)
            all_volatilities.append(volatilities)

        return np.array(all_stock_prices)

    @staticmethod
    def get_weiner_processes_with_correlation(correlation, maturity, time_step_per_maturity=100):
        """
        Gives 2 Brownian motions which are correlated with each other.

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
                    (2, maturity * time_step_per_maturity)
                The rows are the brownian motions and together have correlation of the given value 'correlation'
        """

        dt = 1 / time_step_per_maturity
        number_of_steps = maturity * time_step_per_maturity

        brownian_motions = np.random.randn(2, number_of_steps)
        correlated_bm = np.array([correlation, np.sqrt(1 - correlation ** 2)]).dot(brownian_motions)

        # TODO: controleer op correctheid process (variantie en stapsgrootte)
        # TODO: hoe werkt vstack precies?
        return np.vstack((brownian_motions[0], correlated_bm)) * np.sqrt(dt)

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
                                  seed=42):

        """

        :param amount:
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

        # set seed
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

