from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm


class StockModel(ABC):

    @abstractmethod
    def get_stock_prices(self,
                         amount: int,
                         start_price: float,
                         maturity: int,
                         steps_per_maturity=100,
                         seed=None):
        pass

    def get_price_simulations(self,
                              option_styles,
                              amount_paths: int,
                              start_price: float,
                              maturity: int,
                              interest_rate: float,
                              strike_price=None,
                              option_type='C',
                              steps_per_maturity: int = 100,
                              seed: int = None,
                              max_path_generated: int = None):
        """
        The stock_paths will be generated of the object on which this method is called.

        :param option_styles: A list of class OptionStyle.
        :param amount_paths: Positive integer.
                            This is the total number of paths generated.
        :param start_price: Positive float.
                            Is the starting price of the stock.
        :param maturity: Positive integer.
                        The total time period for the simulation.
                        The period of one payment of the interest_rate should be the same as maturity=1.
        :param interest_rate: a positive value.
                        The interest rate per maturity.
        :param strike_price: Positive float.
                            The price for the stock when the option is exercised.
        :param option_type: 'C' for a call option, 'P' for a put option
                            For both give value ['C','P']
        :param steps_per_maturity: A positive integer. (default = 100)
                                    The amount of small steps taken to represent 1 maturity passing.
                                    The higher the number te more accurate it represents the stock,
                                        but more time consuming
        :param seed:Positive integer or None. (default = None)
                    If value is different from None, the function np.random.seed(seed) will be called.
                    For replication purposes, to get same 'random' values.
        :param max_path_generated: positive int (not 0) or None. (default = None)
                    It is the maximum number of paths that are generated for the option pricing.
                    This is to reduce unnecessary RAM memory, but slower in speed.
                    If value is None, all the paths will be generated at the same time.

        :return: List with all the prices (positive values) of the option_types.
            Same length and order of the prices as the given list option_styles.
            If option_type is list (=['C','P']), then it is a dict with keys 'C' and 'P',
                with resp. the prices of the option_types.
        """

        def partition_maker(total_number: int, value_splitter: int) -> list:
            """
            Function  to make a list for the number of paths that needs to be generated,
                with max value the value_splitter.
            For example, total_number = 16 and value_splitter=5, then the output will be [5, 5, 5, 1].

            :param total_number: int
                                The total value that needs to be split in portions.
            :param value_splitter: int
                                The maximum value the potions may have.
            :return: List with values at most equal to value_splitter.
            """
            deler, rest = divmod(total_number, value_splitter)

            values = [value_splitter] * deler
            if rest != 0:
                values += [rest]

            return values

        if type(max_path_generated) is not int and max_path_generated is not None:
            raise TypeError("max_path_generated must be an integer")

        value_splitter = amount_paths if max_path_generated is None else max_path_generated
        number_paths_generating = partition_maker(amount_paths, value_splitter)

        if seed is not None:
            np.random.seed(seed)

        # if it is not a list, make it a list
        if type(option_styles) is not list:
            option_styles = [option_styles]

        names_options = [str(option) for option in option_styles]

        # dict if option_type is a list
        dict_values_list = {}
        # dict if option_type is 'C' or 'P'
        dict_paths = {}

        # starting to generate a smaller amount of paths, to use less RAM memory
        for n_paths in number_paths_generating:
            # Generation of all the paths.
            simulations = self.get_stock_prices(n_paths,
                                                start_price,
                                                maturity,
                                                steps_per_maturity=steps_per_maturity)

            for option_style in option_styles:
                if type(option_type) is list:
                    for opt_type in option_type:
                        correct_dict_paths = dict_values_list.get(opt_type, dict())

                        prices_option_paths = correct_dict_paths.get(str(option_style), [])
                        new_prices = option_style.get_prices_per_path(simulations,
                                                                      maturity,
                                                                      interest_rate,
                                                                      option_type=opt_type,
                                                                      strike_price=strike_price)
                        prices_option_paths.extend(new_prices)
                        correct_dict_paths[str(option_style)] = prices_option_paths

                        dict_values_list[opt_type] = correct_dict_paths
                else:
                    prices_option_paths = dict_paths.get(str(option_style), [])
                    new_prices = option_style.get_prices_per_path(simulations,
                                                                  maturity,
                                                                  interest_rate,
                                                                  option_type=option_type,
                                                                  strike_price=strike_price)
                    prices_option_paths.extend(new_prices)
                    dict_paths[str(option_style)] = prices_option_paths

        # set everything in the correct structure to return
        if type(option_type) is list:
            for opt_type in option_type:
                prices = [np.mean(dict_values_list.get(opt_type)[name]) for name in names_options]
                dict_values_list[opt_type] = prices
            return dict_values_list
        else:
            prices = [np.mean(dict_paths[name]) for name in names_options]
            return prices


class BlackScholes(StockModel):

    def __init__(self, interest_rate: float, volatility: float):
        """
        :param interest_rate: Positive float.
                            The risk-free interest rate, per time maturity.
        :param volatility: Positive float.
                            The volatility of the stock, per time maturity.
        """
        self.interest_rate = interest_rate
        self.volatility = volatility

    def get_stock_prices(self,
                         amount_paths: int,
                         start_price: float,
                         maturity: int,
                         steps_per_maturity: int = 100,
                         seed: int = None):
        """
        Simulations of stock prices based on the Black Scholes model,
        this means the stock prices follows the Geometric Brownian Motion.

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
        :param seed: Positive integer or None. (default = None)
                    If value is different from None, the function np.random.seed(seed) will be called.
                    For replication purposes, to get same 'random' values.

        :return: 2d numpy.array of all the generated paths, based on the Black Scholes model.
                shape:
                        (amount, maturity * time_step_per_maturity + 1)
                Each row represents a different path, the columns the time.
                The first column is the start_price.
        """
        # set value of seed
        if seed is not None:
            np.random.seed(seed=seed)

        # the amount of timesteps needed for the stockprice
        number_of_steps = maturity * steps_per_maturity
        # length for each step
        dt = 1 / steps_per_maturity

        # calculates a path(process) of factors starting from start_price.
        # Each step is normal distributed with mean 0 and variance dt (length of a step)
        # Because it is a process, i.e. based on the previous value, we take the product of the row ('cumprod(...,1)')
        stock_process = np.cumprod(1 + self.interest_rate * dt +
                                   self.volatility * np.random.normal(0, np.sqrt(dt),
                                                                      (amount_paths, number_of_steps)), 1) * start_price

        # starting prices, for the first column
        first_column = np.ones((amount_paths, 1)) * start_price

        # adding start_price as first element
        stock_process = np.append(first_column, stock_process, axis=1)

        return stock_process

    @staticmethod
    def help_function(start_price: float,
                      strike_price: float,
                      maturity: int,
                      interest_rate: float,
                      volatility: float):
        """
        Help function for the Black Scholes formula. Calculation of an important part of the formula.
        Return d_plus,d_min mostly used in the literature.

        :param maturity: Positive integer or np.array with same length as any other parameters.
                        The time that the option matures.
                        The period of one payment of the interest_rate should be the same as maturity=1.
        :param start_price: Positive float or np.array with same length as any other parameters.
                            Is the starting price of the stock.
        :param strike_price: Positive float or np.array with same length as any other parameters.
                            The price for the stock when the option is exercised.
        :param interest_rate: Positive float or np.array with same length as any other parameters.
                            The risk-free interest rate, per time maturity.
        :param volatility: Positive float or np.array with same length as any other parameters.
                            The volatility of the stock, per time maturity.
        :return: 2 values, d_plus and d_min.
            start_price = S
            strike_price = K
            interest_rate = r
            maturity = T

            d_plus//_min = (log(S/K) + (r +//- (volatility^2)/2) * T) / (volatility * sqrt(T))

        """
        d_plus = (np.log(start_price / strike_price) + (interest_rate + (volatility ** 2) / 2) * maturity) / \
                 (volatility * np.sqrt(maturity))
        d_min = d_plus - volatility * np.sqrt(maturity)

        return d_plus, d_min

    @staticmethod
    def solution_call_option(start_price: float,
                             strike_price: float,
                             maturity: int,
                             interest_rate: float,
                             volatility: float):
        """
            Calculating the call option based on the Black Scholes model.
            The Stock price is the Geometric Brownian Motion
            This assumes the interest rate and volatility are constant during the time of maturity.

            :param maturity: Positive integer or np.array with same length as any other parameters.
                        The time that the option matures.
                        The period of one payment of the interest_rate should be the same as maturity=1.
            :param start_price: Positive float or np.array with same length as any other parameters.
                            Is the starting price of the stock.
            :param strike_price: Positive float or np.array with same length as any other parameters.
                                The price for the stock when the option is exercised.
            :param interest_rate: Positive float or np.array with same length as any other parameters.
                                The risk-free interest rate, per time maturity.
            :param volatility: Positive float or np.array with same length as any other parameters.
                                The volatility of the stock, per time maturity.
            :return: Positive float or np.array of the same length as the input
                    Price of a call-option based on the BS model
            """
        d_plus, d_min = BlackScholes.help_function(start_price, strike_price, maturity, interest_rate, volatility)

        return start_price * norm.cdf(d_plus) - strike_price * np.exp(-interest_rate * maturity) * norm.cdf(d_min)

    @staticmethod
    def solution_put_option(start_price: float,
                            strike_price: float,
                            maturity: float,
                            interest_rate: float,
                            volatility: float):
        """
            Calculating the put option based on the Black Scholes model.
            The Stock price is the Geometric Brownian Motion
            This assumes the interest rate and volatility are constant during the time of maturity.

            :param maturity: Positive integer or np.array with same length as any other parameters.
                        The time that the option matures.
                        The period of one payment of the interest_rate should be the same as maturity=1.
            :param start_price: Positive float or np.array with same length as any other parameters.
                            Is the starting price of the stock.
            :param strike_price: Positive float or np.array with same length as any other parameters.
                                The price for the stock when the option is exercised.
            :param interest_rate: Positive float or np.array with same length as any other parameters.
                                The risk-free interest rate, per time maturity.
            :param volatility: Positive float or np.array with same length as any other parameters.
                                The volatility of the stock, per time maturity.
            :return: Positive float or np.array of the same length as the input
                    Price of a put-option based on the BS model.
            """
        d_plus, d_min = BlackScholes.help_function(start_price, strike_price, maturity, interest_rate, volatility)

        return strike_price * np.exp(-interest_rate * maturity) * norm.cdf(-d_min) - start_price * norm.cdf(-d_plus)

    @staticmethod
    def generate_random_variables(amount: int,
                                  stock_price_bound: tuple,
                                  strike_price_bound: tuple,
                                  maturity_bound: tuple,
                                  interest_rate_bound: tuple,
                                  volatility_bound: tuple,
                                  forward_pricing: bool = False,
                                  seed: int = None):
        """
        Generation of random values for the Back Scholes model.

        :param amount: positive integer, number of random variables generated
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
                                The values will be uniformly selected.
        :param interest_rate_bound: float, tuple or list; only the first 2 elements will be used. (positive values)
                                Bounds where the values of the interest rate will be.
                                The values will be uniformly selected.
        :param volatility_bound: float, tuple or list; only the first 2 elements will be used. (positive values)
                                Bounds where the values of the volatility will be.
                                The values will be uniformly selected.
        :param forward_pricing: bool (default = False)
                                True: the strike prices are based on the percentage of the forward pricing.
                                False: the strike prices are based on the percentage of the start price of the stock.
        :param seed: Positive integer or None. (default = None)
                    If value is different from None, the function np.random.seed(seed) will be called.
                    For replication purposes, to get same 'random' values.

        :return: dict, with keys:
                    "interest_rate", "volatility",   "maturity",
                    "stock_price",   "strike_price", "strike_price_percent",
                    "forward_pricing"
                For each key the values are a np.array of length 'amount' with the random values,
                    but "forward_pricing" is True or False
                        if the percentage of the forward pricing as strike price has been used or not.
        """

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
        if seed is not None:
            np.random.seed(seed=seed)

        # conversion to a tuple, in increasing order and controls if the values are positive.
        stock_price_bound = conversion_and_check(stock_price_bound)
        strike_price_bound = conversion_and_check(strike_price_bound)
        maturity_bound = conversion_and_check(maturity_bound)
        interest_rate_bound = conversion_and_check(interest_rate_bound)
        volatility_bound = conversion_and_check(volatility_bound)

        # Check if the maturity is an integer.
        if type(maturity_bound[0]) is not int or type(maturity_bound[1]) is not int:
            raise ValueError

        # random Integer selection for the maturities
        maturities = np.random.randint(maturity_bound[0], maturity_bound[1], amount)

        # random data selection, Uniform
        interest_rates = np.random.uniform(interest_rate_bound[0], interest_rate_bound[1], amount)
        volatilities = np.random.uniform(volatility_bound[0], volatility_bound[1], amount)
        stock_prices = np.random.uniform(stock_price_bound[0], stock_price_bound[1], amount)
        strike_prices_percentage = np.random.uniform(strike_price_bound[0], strike_price_bound[1], amount)

        # Take a percentage of the stock price.
        # If forward pricing, the strike price is the percentage of the forward price.
        strike_prices = stock_prices * strike_prices_percentage if not forward_pricing \
            else stock_prices * np.exp(interest_rates * maturities) * strike_prices_percentage

        # making dictionary for each parameter
        data_dict = {"interest_rate": interest_rates,
                     "volatility": volatilities,
                     "maturity": maturities,
                     "stock_price": stock_prices,
                     "strike_price": strike_prices,
                     "strike_price_percent": strike_prices_percentage,
                     "forward_pricing": forward_pricing}

        return data_dict


class VarianceGamma(StockModel):

    def __init__(self, interest_rate: float, volatility: float, skewness: float, kurtosis: float):
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
        self.interest_rate = interest_rate
        self.skewness = skewness
        if volatility < 0:
            raise ValueError("The volatility must be a positive value")
        if kurtosis < 0:
            raise ValueError("The kurtosis must be a positive value")
        self.volatility = volatility
        self.kurtosis = kurtosis

    def get_stock_prices(self,
                         amount_paths: int,
                         start_price: float,
                         maturity: int,
                         steps_per_maturity: int = 100,
                         seed: int = None):
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

    def variance_process(self,
                         amount_paths: int,
                         maturity: int = 1,
                         steps_per_maturity: int = 100):
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

    def variance_process_brownian_motion(self,
                                         amount_paths: int,
                                         maturity: int = 1,
                                         steps_per_maturity: int = 100):
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
    def generate_random_variables(amount: int,
                                  stock_price_bound: tuple,
                                  strike_price_bound: tuple,
                                  maturity_bound: tuple,
                                  interest_rate_bound: tuple,
                                  skewness_bound: tuple,
                                  volatility_bound: tuple,
                                  kurtosis_bound: tuple,
                                  forward_pricing: bool = False,
                                  seed: int = None):
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


class HestonModel(StockModel):

    def __init__(self,
                 interest_rate: float,
                 start_volatility: float,
                 long_volatility: float,
                 rate_revert_to_long: float,
                 volatility_of_volatility: float,
                 correlation_processes: float):
        """
        :param interest_rate: A positive value.
                            For the fixed interest rate for the Heston model.
        :param start_volatility: A positive value.
                                The starting position of the volatility.
        :param long_volatility: A positive value.
                            The value of volatility which the volatility converges to.
        :param rate_revert_to_long: A positive value.
                                    The 'speed' on which the volatility converges to the long_volatility.
        :param volatility_of_volatility: A positive value.
                                The volatility of the volatility. (same concept as the volatility of the stock price)
        :param correlation_processes: A number between [-1, 1] for the correlation between the brownian motions,
                                    i.e. the brownian motions for the stock prices and the volatility.
        """
        self.start_volatility = start_volatility
        self.interest_rate = interest_rate
        self.long_variance = long_volatility
        self.rate_revert_to_long = rate_revert_to_long
        self.volatility_of_volatility = volatility_of_volatility

        if correlation_processes < -1 or correlation_processes > 1:
            raise ValueError("Incorrect correlation")
        self.correlation_processes = correlation_processes

    def get_stock_prices(self,
                         n_paths: int,
                         start_price: float,
                         maturity: int,
                         steps_per_maturity: int = 100,
                         seed: int = None):
        """
        Simulations of stock prices based on the Heston model.

        :param n_paths: Positive integer.
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
        :param seed: Positive integer. (default = 42)
                    For replication purposes, to get same 'random' values.
        :return: 2d numpy.array of all the generated paths and volatilities, based on the Heston model.
                shape:
                       (2, (amount, maturity * time_step_per_maturity + 1))
                First element is the generated paths, the second the generated volatilities (positive or 0) of each path
                Each row represents a different path, the columns the time.
                The first column is the start_price.

        """

        if seed is not None:
            np.random.seed(seed=seed)

        dt = 1 / steps_per_maturity
        number_of_steps = maturity * steps_per_maturity

        # get the full process of the correlated Brownian motions
        all_processes = self.get_brownian_motions_with_correlation(n_paths,
                                                                   self.correlation_processes,
                                                                   maturity,
                                                                   steps_per_maturity=steps_per_maturity)
        # set names for the different processes.
        brownianmotion_stock_processes = all_processes[:, 0, :]
        brownianmotion_volatility_processes = all_processes[:, 1, :]

        # make matrix for all the volatilities
        all_volatilities = np.zeros((n_paths, number_of_steps))
        # first position set to start volatility
        all_volatilities[:, 0] = self.start_volatility

        # function for each iteration per timestep
        def func_vol(position, step_size, volatilities, weiner_volatility):
            # get last volatility's of each path
            last_vol = volatilities[:, position]
            # set negative values to 0
            not_negative_vol = last_vol.copy()
            not_negative_vol[not_negative_vol < 0] = 0

            dnu = self.rate_revert_to_long * (self.long_variance - not_negative_vol) * step_size + \
                  self.volatility_of_volatility * np.sqrt(not_negative_vol) * weiner_volatility[:, position]

            return dnu

        # lambda expression to get array of the volatility's for each path
        array_f = lambda pos, step_size, vola, process: np.asarray(func_vol(pos, step_size, vola, process))

        # Calculate simultaneously (all paths) the next step.
        for i in range(number_of_steps - 1):
            all_volatilities[:, i + 1] = all_volatilities[:, i] + array_f(i, dt, all_volatilities,
                                                                          brownianmotion_volatility_processes)

        # Because of the Euler scheme, it is possible to get negative values for Volatility.
        # This cause problems because the square root is taken for the volatility.
        all_volatilities[all_volatilities < 0] = 0

        # start of the Stock model process
        rate = self.interest_rate * dt
        vol = np.sqrt(all_volatilities) * brownianmotion_stock_processes

        # taking the product of the previous elements til the end.
        # not take first element of 'vol' because this is already 0, otherwise it wil get one time to much interest
        total_process = np.cumprod(1 + rate + vol, 1)

        first_column_stock = np.ones((n_paths, 1))
        total_process = np.append(first_column_stock, total_process, axis=1)
        total_process *= start_price

        return total_process

    def get_stock_prices_v2(self,
                            n_paths: int,
                            start_price: float,
                            maturity: int,
                            steps_per_maturity: int = 100,
                            seed: int = None):
        if seed is not None:
            np.random.seed(seed=seed)

        dt = 1 / steps_per_maturity
        number_of_steps = maturity * steps_per_maturity

        all_volatilities = []

        all_processes = self.get_brownian_motions_with_correlation(n_paths,
                                                                   self.correlation_processes,
                                                                   maturity,
                                                                   steps_per_maturity=steps_per_maturity)

        brownianmotion_stock_processes = all_processes[:, 0, :]
        brownianmotion_volatility_processes = all_processes[:, 1, :]

        # start of process of the volatility
        for j in range(n_paths):
            volatilities = [self.start_volatility]
            bm_volatility = brownianmotion_volatility_processes[j]

            # we don't need the last element of the volatilities (Euler method)
            for i in range(number_of_steps - 1):
                last_vol = volatilities[-1]
                not_negative_vol = max(last_vol, 0)

                dnu = self.rate_revert_to_long * (self.long_variance - not_negative_vol) * dt + \
                      self.volatility_of_volatility * np.sqrt(not_negative_vol) * bm_volatility[i]
                volatilities.append(last_vol + dnu)
            all_volatilities.append(volatilities)
        # change it to a numpy array.
        all_volatilities = np.array(all_volatilities)

        # Because of the Euler scheme, it is possible to get negative values for Volatitlity.
        # This cause problems because the square root is taken for the volatility.
        all_volatilities[all_volatilities < 0] = 0

        rate = self.interest_rate * dt
        vol = np.sqrt(all_volatilities) * brownianmotion_stock_processes

        # taking the product of the previous elements til the end.
        # not take first element of 'vol' because this is already 0, otherwise it wil get one time to much interest
        total_process = np.cumprod(1 + rate + vol, 1)

        first_column_stock = np.ones((n_paths, 1))
        total_process = np.append(first_column_stock, total_process, axis=1)
        total_process *= start_price

        return total_process, all_volatilities

    def get_stock_prices_naive_simulation_v1(self,
                                             n_paths: int,
                                             start_price: float,
                                             maturity: int,
                                             steps_per_maturity: int = 100,
                                             seed: int = None):
        if seed is not None:
            np.random.seed(seed=seed)

        dt = 1 / steps_per_maturity
        number_of_steps = maturity * steps_per_maturity

        all_stock_prices = []
        all_volatilities = []

        all_processes = self.get_brownian_motions_with_correlation(n_paths,
                                                                   self.correlation_processes,
                                                                   maturity,
                                                                   steps_per_maturity=steps_per_maturity)

        brownianmotion_stock_processes = all_processes[:, 0, :]
        brownianmotion_volatility_processes = all_processes[:, 1, :]

        for j in range(n_paths):
            stock_prices = [start_price]
            volatilities = [self.start_volatility]
            bm_stock = brownianmotion_stock_processes[j]
            bm_volatility = brownianmotion_volatility_processes[j]

            for i in range(number_of_steps):
                last_price = stock_prices[-1]
                last_vol = volatilities[-1]
                not_negative_vol = max(last_vol, 0)

                dS = last_price * (self.interest_rate * dt + np.sqrt(not_negative_vol) * bm_stock[i])
                dnu = self.rate_revert_to_long * (self.long_variance - not_negative_vol) * dt + \
                      self.volatility_of_volatility * \
                      np.sqrt(not_negative_vol) * bm_volatility[i]

                # adding the next stock prices and volatilities
                stock_prices.append(stock_prices[-1] + dS)
                volatilities.append(volatilities[-1] + dnu)
            all_stock_prices.append(stock_prices)
            all_volatilities.append(volatilities)

        return np.array(all_stock_prices), np.array(all_volatilities)

    @staticmethod
    def get_brownian_motions_with_correlation(n_paths: int,
                                              correlation: float,
                                              maturity: int,
                                              steps_per_maturity: int = 100):
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
        :param steps_per_maturity: A positive integer. (default = 100)
                                    The amount of small steps taken to represent 1 maturity passing.
                                    The higher the number te more accurate it represents the stock,
                                        but more time consuming
        :return: 2d numpy.array with shape:
                    (n_paths, 2, maturity * time_step_per_maturity)
                The rows are the brownian motions and together have correlation of the given value 'correlation'
        """

        dt = 1 / steps_per_maturity
        number_of_steps = maturity * steps_per_maturity

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
    def generate_random_variables(amount: int,
                                  stock_price_bound: tuple,
                                  strike_price_bound: tuple,
                                  maturity_bound: tuple,
                                  interest_rate_bound: tuple,
                                  start_vol_bound: tuple,
                                  long_volatility_bound: tuple,
                                  rate_revert_to_long_bound: tuple,
                                  vol_of_vol_bound: tuple,
                                  correlation_bound: tuple,
                                  forward_pricing: bool = False,
                                  seed: int = None):

        """
        The generation of random variables for the Heston Model.

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
        :param start_vol_bound: float, tuple or list; only the first 2 elements will be used. (positive values)
                                Bounds where the values of the start volatility will be.
                                The values will be uniformly selected.
        :param long_volatility_bound: float, tuple or list; only the first 2 elements will be used. (positive values)
                                Bounds where the values of the 'long volatility' will be.
                                The values will be uniformly selected.
        :param rate_revert_to_long_bound: tuple or list; only the first 2 elements will be used. (positive values)
                                Bounds where the values of the rate to revert to the long volatility will be.
                                The values will be uniformly selected.
        :param vol_of_vol_bound:tuple or list; only the first 2 elements will be used. (positive values)
                                Bounds where the values of the volatility of the volatility will be.
                                The values will be uniformly selected.
        :param correlation_bound: tuple or list; only the first 2 elements will be used. Values in the interval [-1,1].
                                Bounds where the values of the correlation between the brownian motions will be.
                                The values will be uniformly selected.
        :param forward_pricing: bool (default = False)
                                True: the strike prices are based on the percentage of the forward pricing.
                                False: the strike prices are based on the percentage of the start price of the stock.
        :param seed: Positive integer or None. (default = None)
                    If value is different from None, the function np.random.seed(seed) will be called.
                    For replication purposes, to get same 'random' values.
        :return: dict with keys
                    "stock_price", "strike_price", "strike_price_percent", "interest_rate",
                    "maturity",    "start_vol",    "long_volatility",      "rate_revert_to_long",
                    "vol_of_vol",  "correlation",
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

        # set seed if different from None.
        if seed is not None:
            np.random.seed(seed=seed)

        # conversion to a tuple, in increasing order and controls if the values are positive.
        stock_price_bound = conversion_and_check(stock_price_bound)
        strike_price_bound = conversion_and_check(strike_price_bound)
        maturity_bound = conversion_and_check(maturity_bound)
        interest_rate_bound = conversion_and_check(interest_rate_bound)
        start_vol_bound = conversion_and_check(start_vol_bound)
        long_volatility_bound = conversion_and_check(long_volatility_bound)
        rate_revert_to_long_bound = conversion_and_check(rate_revert_to_long_bound)
        vol_of_vol_bound = conversion_and_check(vol_of_vol_bound)

        correlation_bound = conversion_and_check(correlation_bound, check_positive=False)

        # Check if the maturity is an integer.
        if type(maturity_bound[0]) is not int or type(maturity_bound[1]) is not int:
            raise ValueError

        # Check if correlation is between -1 and 1
        if correlation_bound[0] < -1 or correlation_bound[1] > 1:
            raise ValueError("Values of correlation must be between -1 and 1")

        # random Integer selection for the maturities
        maturities = np.random.randint(maturity_bound[0], maturity_bound[1], amount)

        # random data selection, Uniform
        stock_prices = np.random.uniform(stock_price_bound[0], stock_price_bound[1], amount)
        strike_prices_percentage = np.random.uniform(strike_price_bound[0], strike_price_bound[1], amount)
        interest_rates = np.random.uniform(interest_rate_bound[0], interest_rate_bound[1], amount)
        start_vols = np.random.uniform(start_vol_bound[0], start_vol_bound[1], amount)
        long_variances = np.random.uniform(long_volatility_bound[0], long_volatility_bound[1], amount)
        rate_revert_to_longs = np.random.uniform(rate_revert_to_long_bound[0], rate_revert_to_long_bound[1], amount)
        vol_of_vols = np.random.uniform(vol_of_vol_bound[0], vol_of_vol_bound[1], amount)
        correlations = np.random.uniform(correlation_bound[0], correlation_bound[1], amount)

        # Take a percentage of the stock price
        strike_prices = stock_prices * strike_prices_percentage if not forward_pricing \
            else stock_prices * np.exp(interest_rates * maturities) * strike_prices_percentage

        # making dictionary for each parameter
        data_dict = {"stock_price": stock_prices,
                     "strike_price": strike_prices,
                     "strike_price_percent": strike_prices_percentage,
                     "interest_rate": interest_rates,
                     "maturity": maturities,
                     "start_vol": start_vols,
                     "long_volatility": long_variances,
                     "rate_revert_to_long": rate_revert_to_longs,
                     "vol_of_vol": vol_of_vols,
                     "correlation": correlations,
                     "forward_pricing": forward_pricing}

        return data_dict
