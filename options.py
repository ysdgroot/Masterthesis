from abc import ABC, abstractmethod
import math
import numpy as np


class Option(ABC):

    def __init__(self):
        # sets a dict for the class OptionStyle for the standard functionality of the call and put option
        self.optiontype_dict = dict({"C": self.call, "P": self.put})

    @staticmethod
    def call(stock_price, strike_price):
        """
        The value of a call option (buying stock),
        which has value (stock_price - strike_price)    if stock_price > strike price
                        0                               if stock_price < strike_price
        :param stock_price: a value or numpy.array of numbers.
                            If numpy.array, the length must be equal to the length of the strike_price array.
        :param strike_price: a value or numpy.array of numbers.
                            If numpy.array, the length must be equal to the length of the stock_price array.
        :return: value or numpy.array (same shape) with positive values or zero
        """
        return (stock_price - strike_price) * (stock_price > strike_price)

    @staticmethod
    def put(stock_price, strike_price):
        """
        The value of a put option (selling stock),
        which has value (strike_price - stock_price) if stock_price < strike_price
                        0                            if stock_price > strike_price
        :param stock_price: a value or numpy.array of numbers
                            If numpy.array, the length must be equal to the length of the strike_price array.
        :param strike_price: a value or numpy.array of numbers
                             If numpy.array, the length must be equal to the length of the stock_price array.
        :return: value or numpy.array (same shape) with positive values or zero
        """
        return (strike_price - stock_price) * (strike_price > stock_price)

    def get_dict(self):
        """
        Function to get the dictionary for the different types of option_types, namely call en put option_types.
        "C" stands for the call option
        "P" stands for the put option
        :return: dict with keys "C" and "P" with values the corresponding function
        """
        return dict({"C": self.call, "P": self.put})

    def get_price_option(self, stock_paths,
                         maturity,
                         interest_rate,
                         option_type="C",
                         strike_price=None):
        """
        A method necessary for this class, to price the option based on the given paths.

        This function is for simulations, to approximate the optimal option price.
        The optimal price is the mean of the possible profits (zero if there is no profit).

        :param stock_paths: 2d numpy.array with each row a stock path (price)
                            The columns represents the price at the time for the stock.
                            First column is the same value, which is the start_price. (There is no check for this)
        :param maturity: The time of maturity of the option.
                        Necessary for the price under the risk natural measure.
        :param interest_rate: The interest_rate per value of maturity.
                        Necessary for the price under the risk natural measure.
        :param option_type: 'C' for a call option (default value)
                            'P' for a put option
        :param strike_price: a positive number or None. (default=None)
                            For generalisation purposes is the default value None.

        :return: A positive value, which represents the price of the option.
        """

        # Get the price (now = discounted price)
        option_values = self.get_prices_per_path(stock_paths,
                                                 maturity,
                                                 interest_rate,
                                                 option_type=option_type,
                                                 strike_price=strike_price)

        # Price of the option in the value of all option_values
        option_price = np.mean(option_values)

        # The variance of all the pricing_values (which are already discounted)
        var_option_pricing = np.var(option_values)

        return option_price, var_option_pricing

    def get_prices_per_path(self, stock_paths, maturity, interest_rate, option_type="C", strike_price=None):
        """
        This function is for simulations, to approximate the optimal option price.
        The optimal price is the mean of the possible profits (zero if there is no profit).

        :param stock_paths: 2d numpy.array with each row a stock path (price)
                            The columns represents the price at the time for the stock.
                            First column is the same value, which is the start_price. (There is no check for this)
        :param maturity: The time of maturity of the option.
                        Necessary for the price under the risk natural measure.
        :param interest_rate: The interest_rate per value of maturity.
                        Necessary for the price under the risk natural measure.
        :param option_type: 'C' for a call option (default value)
                            'P' for a put option
        :param strike_price: a positive float or None. (default=None)
                            If None, the default value will be the first price starting of the paths,
                                unless the strike price of the option is path-dependent,
                                in this case the value of strike_price will not be used, but instead be calculated.

        :return: A positive value, which represents the price of the option at the moment (risk natural measure)
        """

        # check if the option_type is correct
        option_function = self.optiontype_dict.get(option_type)
        if option_function is None:
            raise ValueError("Invalid option_type")

        strike_prices = self.get_all_strike_prices(stock_paths, strike_price)

        # Gets the last value of the stock, for European vanilla option (the last column)
        stock_prices = stock_paths[:, -1]

        # the price under the risk natural measure
        option_price_paths = math.e ** (-maturity * interest_rate) * option_function(stock_prices, strike_prices)

        return option_price_paths

    @abstractmethod
    def get_all_strike_prices(self, stock_paths, strike_price):
        """
        Function to generalise the option pricing.
        """
        pass


class AsianMean(Option):

    def __init__(self):
        super(AsianMean, self).__init__()

    def get_all_strike_prices(self, stock_paths, strike_price):
        """
        :param stock_paths: 2d numpy.array with each row a stock path (price)
                            The columns represents the price at the time for the stock.
                            First column is the same value, which is the start_price. (There is no check for this)
        :param strike_price: this value will be ignored

        :return: a 1D np.array of length the number of generated paths (= number of columns stock_paths)
                Each value is the mean of the stock prices.
        """
        # The strike_prices by the mean of all the paths
        strike_prices = np.mean(stock_paths, axis=1)

        return strike_prices

    def __str__(self):
        return "AsianMean"


class Lookback(Option):

    def __init__(self, lookback_min=True):
        """
        Object to represent the lookback option in European style in a general form.

        The strike price of the lookback option is based on the minimum or maximum value that the stock has taken.
        :param lookback_min: boolean (default = True)
                            If the function which the lookback is based on the minimum value or the maximum value.
        """
        super(Lookback, self).__init__()
        self.lookback_min = lookback_min

    def get_all_strike_prices(self, stock_paths, strike_price):
        """
        :param stock_paths: 2d numpy.array with each row a stock path (price)
                            The columns represents the price at the time for the stock.
                            First column is the same value, which is the start_price. (There is no check for this)
        :param strike_price: this value will be ignored

        :return: a 1D np.array of length the number of generated paths (= number of columns stock_paths)
                Each value is the min or max (depends on the class value) of the stock prices.
        """

        strike_prices = np.min(stock_paths, axis=1) if self.lookback_min else np.max(stock_paths, axis=1)
        return strike_prices

    def __str__(self):
        str_value = "min" if self.lookback_min else "max"
        return f"Lookback_{str_value}"


class PlainVanilla(Option):

    def __init__(self):
        super(PlainVanilla, self).__init__()

    def get_all_strike_prices(self, stock_paths, strike_price):
        """
        Function to generalise the option pricing

        :param stock_paths: 2d numpy.array with each row a stock path (price)
                            The columns represents the price at the time for the stock.
                            First column is the same value, which is the start_price. (There is no check for this)
        :param strike_price: a positive float or None.
                            In case of None, the strike price will be the first value of the stock_paths,
                            in the other case the value will be strike_price itself

        :return: a single float
        """
        strike_prices = strike_price if strike_price is not None else stock_paths[0, 0]

        return strike_prices

    def __str__(self):
        return "Plainvanilla"
