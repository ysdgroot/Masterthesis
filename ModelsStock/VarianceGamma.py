from ModelsStock.GeneralStockModel import StockModel
import numpy as np


class VarianceGamma(StockModel):

    # TODO: geef andere namen aan skewness, volatility en kurtosis, om een beter beeld te geven wat die betekenen.
    def __init__(self, interest_rate, volatility, skewness, kurtosis):
        """

        :param interest_rate:Positive float.
                            The risk-free interest rate, per time maturity.
        :param skewness: float.
                    The implied skewness for the Variance Gamma process.
        :param volatility: Positive float.
                    The implied volatility for the Variance Gamma process.
        :param kurtosis: Positive float.
                    The implied kurtosis for the Variance Gamma process.
        """
        print("Controleer op volgordes van de VG model, dit is aangepast!!")
        self.interest_rate = interest_rate
        self.skewness = skewness
        if volatility < 0:
            raise ValueError("The volatility must be a positive value")
        if kurtosis < 0:
            raise ValueError("The kurtosis must be a positive value")
        self.volatility = volatility
        self.kurtosis = kurtosis

    def get_stock_prices(self, amount_paths, start_price, maturity, steps_per_maturity=100, seed=None):
        """
        Simulations of stock prices based on the Variance Gamma model.

        :param amount_paths: Positive integer.
                            This is the total number of paths generated.
        :param start_price: Positive float.
                            Is the starting price of the stock.
        :param maturity: Positive integer.
                        The total time period for the simulation.
                        The period of one payment of the interest_rate should be the same as maturity=1.
        :param steps_per_maturity: A positive integer. (default = 100)
                                    The amount of small steps taken to represent 1 maturity passing.
                                    The higher the number te more accurate it represents the stock,
                                        but more time consuming
        :param seed: Positive integer. (default = None)
                    For replication purposes, to get same 'random' values.
        :return: 2d numpy.array of all the generated paths, based on Variance Gamma model.
                shape:
                        (amount, maturity * time_step_per_maturity + 1)
                Each row represents a different path, the columns the time.
                The first column is the start_price.
        """
        if seed is not None:
            np.random.seed(seed=seed)

        number_of_evaluations = steps_per_maturity * maturity
        dt = 1 / steps_per_maturity

        # omega based on the article
        omega = np.log(1 - self.skewness * self.kurtosis - self.kurtosis * self.volatility ** 2 / 2) / self.kurtosis

        # the process based on the variance gamma model, each increment or decrement for each time_step
        # variance_process = self.variance_process(amount,
        #                                          time_step_per_maturity=time_step_per_maturity,
        #                                          maturity=maturity, seed=seed)

        # This test is faster than the 'variance_process' function.
        variance_process = self.variance_process_brownian_motion(amount_paths,
                                                                 steps_per_maturity=steps_per_maturity,
                                                                 maturity=maturity)

        # Start with the 0 on position 0 (so S_t=0 = S0)
        constant_rate_stock = np.cumsum(np.append(0, np.full(number_of_evaluations - 1,
                                                             (self.interest_rate + omega) * dt)))

        # The stock price on time t, based on the variance gamma
        total_exponent = np.add(variance_process, constant_rate_stock)

        # Adding 0 in the first column, so the first column (first value of the paths) will be the start price
        first_column = np.zeros((amount_paths, 1))
        total_exponent = np.append(first_column, total_exponent, axis=1)

        return start_price * np.exp(total_exponent)

    def variance_process(self, amount_paths, maturity=1, steps_per_maturity=100):
        """
        Creates a sequence of numbers that represents the Gamma Variance process based on 2 Variance Gamma processes
        This function is bit slower than the 'variance_process_brownian_motion' function

        Based on the paper "Variance-Gamma and Monte Carlo" from Michael C. Fu

        :param amount_paths: Positive integer.
                            This is the total number of paths generated.
        :param maturity: Positive integer.
                        The total time period for the simulation.
                        The period of one payment of the interest_rate should be the same as maturity=1.
        :param steps_per_maturity: A positive integer. (default = 100)
                                    The amount of small steps taken to represent 1 maturity passing.
                                    The higher the number te more accurate it represents the stock,
                                        but more time consuming
        :return: 2d numpy.array, with the Variance Gamma process.
                shape:
                        (amount, maturity * time_step_per_maturity)
        """

        number_of_steps = maturity * steps_per_maturity
        size_increments = 1 / steps_per_maturity

        # declaration of values for the gamma process. (Just for readability reason)
        mu_plus = (np.sqrt(self.skewness ** 2 + 2 * self.volatility ** 2 / self.kurtosis) + self.skewness) / 2
        mu_min = (np.sqrt(self.skewness ** 2 + 2 * self.volatility ** 2 / self.kurtosis) - self.skewness) / 2

        # Variance Gamma process is the difference of 2 Gamma processes
        gamma_process_plus = np.random.gamma(size_increments / self.kurtosis, self.kurtosis * mu_plus,
                                             (number_of_steps, amount_paths))
        gamma_process_min = np.random.gamma(size_increments / self.kurtosis, self.kurtosis * mu_min,
                                            (number_of_steps, amount_paths))

        # todo: draaien van het process
        # verwijder transpose, verander axis=1, en verwissel n_steps en amount_paths
        return np.cumsum(gamma_process_plus - gamma_process_min, axis=0).transpose()

    def variance_process_brownian_motion(self, amount_paths, maturity=1, steps_per_maturity=100):
        """
        Creates a sequence of numbers that represents the Gamma Variance process based on Brownian motion.
        With a standard normal distribution in the process.
        This is a faster method than the process based on the difference of 2 gamma distributed sequences.

        Based on the paper "Variance-Gamma and Monte Carlo" from Michael C. Fu

        :param amount_paths: Positive integer.
                            This is the total number of paths generated.
        :param maturity: Positive integer.
                        The total time period for the simulation.
                        The period of one payment of the interest_rate should be the same as maturity=1.
        :param steps_per_maturity: A positive integer. (default = 100)
                                    The amount of small steps taken to represent 1 maturity passing.
                                    The higher the number te more accurate it represents the stock,
                                        but more time consuming
        :return: 2d numpy.array, with the Variance Gamma process.
                shape:
                        (amount, maturity * time_step_per_maturity)
        """
        number_of_steps = maturity * steps_per_maturity
        size_increments = 1 / steps_per_maturity

        # Variance Gamma process which is based on a Brownian motion
        gamma_process = np.random.gamma(size_increments / self.kurtosis, self.kurtosis, (number_of_steps, amount_paths))
        brownian_motion = np.random.randn(number_of_steps, amount_paths)

        return np.cumsum(self.skewness * gamma_process + self.volatility * np.sqrt(gamma_process) * brownian_motion,
                         axis=0).transpose()

    @staticmethod
    def generate_random_variables(amount,
                                  stock_price_bound,
                                  strike_price_bound,
                                  maturity_bound,
                                  interest_rate_bound,
                                  theta_bound,
                                  sigma_bound,
                                  nu_bound,
                                  forward_pricing=False,
                                  seed=None):
        """

        :param amount:
        :param stock_price_bound:
        :param strike_price_bound:
        :param maturity_bound:
        :param interest_rate_bound:
        :param theta_bound:
        :param sigma_bound:
        :param nu_bound:
        :param seed:
        :return:
        """
        # todo: schrijven van documentatie

        def conversion_and_check(value, check_positive=True):
            """
            Convert the values in tuples, so it makes it easier to use.
            :param value: single value, tuple or list.
                        A single value will be converted in a 2-tuple.
                        In case it is a tuple/list with 3 or more values, only the first 2 elements will be used.
            :param check_positive: boolean (default=True)
                        True if the value must be a positive value
                        False if it not necessary for a positive value
            :return: a tuple in the correct format.
                    If positive_value = True and if 'value' contains a negative number, a ValueError will be raised.
            """
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
            if check_positive and bounds[0] < 0:
                raise ValueError
            return bounds

        # set seed
        if seed is not None:
            np.random.seed(seed=seed)

        # conversion to a tuple, in increasing order and controls if the values are positive.
        stock_price_bound = conversion_and_check(stock_price_bound)
        strike_price_bound = conversion_and_check(strike_price_bound)
        maturity_bound = conversion_and_check(maturity_bound)
        interest_rate_bound = conversion_and_check(interest_rate_bound)
        theta_bound = conversion_and_check(theta_bound, check_positive=False)
        sigma_bound = conversion_and_check(sigma_bound)
        nu_bound = conversion_and_check(nu_bound)

        # Check if the maturity is an integer.
        if type(maturity_bound[0]) is not int or type(maturity_bound[1]) is not int:
            raise ValueError

        # random Integer selection for the maturities
        maturities = np.random.randint(maturity_bound[0], maturity_bound[1], amount)

        # random data selection, Uniform
        stock_prices = np.random.uniform(stock_price_bound[0], stock_price_bound[1], amount)
        strike_prices_percentage = np.random.uniform(strike_price_bound[0], strike_price_bound[1], amount)
        interest_rates = np.random.uniform(interest_rate_bound[0], interest_rate_bound[1], amount)
        thetas = np.random.uniform(theta_bound[0], theta_bound[1], amount)
        sigmas = np.random.uniform(sigma_bound[0], sigma_bound[1], amount)
        nus = np.random.uniform(nu_bound[0], nu_bound[1], amount)

        # Take a percentage of the stock price
        strike_prices = stock_prices * strike_prices_percentage if not forward_pricing \
            else stock_prices * np.exp(interest_rates * maturities) * strike_prices_percentage

        # Making dictionary for each parameter
        data_dict = {"stock_price": stock_prices,
                     "strike_price": strike_prices,
                     "interest_rate": interest_rates,
                     "strike_price_percent": strike_prices_percentage,
                     "maturity": maturities,
                     "skewness": thetas,
                     "volatility": sigmas,
                     "kurtosis": nus,
                     "forward_pricing": forward_pricing}

        return data_dict
