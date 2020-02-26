from ModelsStock.GeneralStockModel import StockModel
import numpy as np


class VarianceGamma(StockModel):

    def __init__(self, interest_rate, volatility, skewness, kurtosis):
        """

        :param interest_rate:Positive float.
                            The risk-free interest rate, per time maturity.
        :param skewness: (parameter 'theta') float.
                    The implied skewness for the Variance Gamma process.
        :param volatility: (parameter 'sigma') Positive float.
                    The implied volatility for the Variance Gamma process.
        :param kurtosis: (parameter 'nu') Positive float.
                    The implied kurtosis for the Variance Gamma process.
        """
        # todo controleer op volgordes
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
        # todo geef referentie voor meer uitleg erover

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
                                             (amount_paths, number_of_steps))
        gamma_process_min = np.random.gamma(size_increments / self.kurtosis, self.kurtosis * mu_min,
                                            (amount_paths, number_of_steps))

        return np.cumsum(gamma_process_plus - gamma_process_min, axis=1)

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
        gamma_process = np.random.gamma(size_increments / self.kurtosis, self.kurtosis, (amount_paths, number_of_steps))
        brownian_motion = np.random.randn(amount_paths, number_of_steps)

        return np.cumsum(self.skewness * gamma_process + self.volatility * np.sqrt(gamma_process) * brownian_motion,
                         axis=1)

    @staticmethod
    def generate_random_variables(amount,
                                  stock_price_bound,
                                  strike_price_bound,
                                  maturity_bound,
                                  interest_rate_bound,
                                  skewness_bound,
                                  volatility_bound,
                                  kurtosis_bound,
                                  forward_pricing=False,
                                  seed=None):
        """

        The generation of random variables for the Variance Gamma model.

        :param amount: A positive integer.
                        The amount of different values, of each parameter.
        :param stock_price_bound: float, tuple or list; only the first 2 elements will be used. (positive values)
                                Bounds where the values of the stock prices will be.
                                The values will be uniformly selected.
        :param strike_price_bound: float, tuple or list; only the first 2 elements will be used. (positive values)
                                Bounds are the the strike prices, but are the percentages(!) of the stock_price.
                                The values will be uniformly selected.
                                (!) If forward_pricing = True, then the strike_prices are the percentage
                                    of the forward pricing (=e^(r*T)*S0)
        :param maturity_bound: int, tuple or list; only the first 2 elements will be used. (positive values)
                                Bounds where the values of the maturity will be.
                                The values will be uniformly selected. (uniformly over the int)
        :param interest_rate_bound: float, tuple or list; only the first 2 elements will be used. (positive values)
                                Bounds where the values of the interest rate will be.
                                The values will be uniformly selected.
        :param skewness_bound: float, tuple or list; only the first 2 elements will be used.
                                Bounds where the values of the skewness will be.
                                The values will be uniformly selected.
        :param volatility_bound:float, tuple or list; only the first 2 elements will be used. (positive values)
                                Bounds where the values of the start volatility will be.
                                The values will be uniformly selected.
        :param kurtosis_bound:float, tuple or list; only the first 2 elements will be used. (positive values)
                                Bounds where the values of the kurtosis will be.
                                The values will be uniformly selected.
        :param forward_pricing: bool (default = False)
                                True: the strike prices are based on the percentage of the forward pricing.
                                False: the strike prices are based on the percentage of the start price of the stock.
        :param seed: Positive integer or None. (default = None)
                    If value is different from None, the function np.random.seed(seed) will be called.
                    For replication purposes, to get same 'random' values.

        "stock_price": stock_prices,
                     "strike_price": strike_prices,
                     "interest_rate": interest_rates,
                     "strike_price_percent": strike_prices_percentage,
                     "maturity": maturities,
                     "skewness": skewness,
                     "volatility": volatilities,
                     "kurtosis": kurtosis,
                     "forward_pricing": forward_pricing}

        :return:dict with keys
                    "stock_price", "strike_price", "strike_price_percent", "interest_rate",
                    "maturity",    "skewness",      "volatility",           "kurtosis",
                    "forward_pricing"
                For each key the values are a np.array of length 'amount' with the random values,
                    but "forward_pricing" is True or False
                        if the percentage of the forward pricing as strike price has been used or not.
        """

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
        skewness_bound = conversion_and_check(skewness_bound, check_positive=False)
        volatility_bound = conversion_and_check(volatility_bound)
        kurtosis_bound = conversion_and_check(kurtosis_bound)

        # Check if the maturity is an integer.
        if type(maturity_bound[0]) is not int or type(maturity_bound[1]) is not int:
            raise ValueError

        # random Integer selection for the maturities
        maturities = np.random.randint(maturity_bound[0], maturity_bound[1], amount)

        # random data selection, Uniform
        stock_prices = np.random.uniform(stock_price_bound[0], stock_price_bound[1], amount)
        strike_prices_percentage = np.random.uniform(strike_price_bound[0], strike_price_bound[1], amount)
        interest_rates = np.random.uniform(interest_rate_bound[0], interest_rate_bound[1], amount)
        skewness = np.random.uniform(skewness_bound[0], skewness_bound[1], amount)
        volatilities = np.random.uniform(volatility_bound[0], volatility_bound[1], amount)
        kurtosis = np.random.uniform(kurtosis_bound[0], kurtosis_bound[1], amount)

        # Take a percentage of the stock price
        strike_prices = stock_prices * strike_prices_percentage if not forward_pricing \
            else stock_prices * np.exp(interest_rates * maturities) * strike_prices_percentage

        # Making dictionary for each parameter
        data_dict = {"stock_price": stock_prices,
                     "strike_price": strike_prices,
                     "interest_rate": interest_rates,
                     "strike_price_percent": strike_prices_percentage,
                     "maturity": maturities,
                     "skewness": skewness,
                     "volatility": volatilities,
                     "kurtosis": kurtosis,
                     "forward_pricing": forward_pricing}

        return data_dict
