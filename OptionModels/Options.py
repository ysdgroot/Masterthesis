from abc import ABC, abstractmethod


class OptionStyle(ABC):

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
        Function to get the dictionary for the different types of options, namely call en put options.
        "C" stands for the call option
        "P" stands for the put option
        :return: dict with keys "C" and "P" with values the corresponding function
        """
        return dict({"C": self.call, "P": self.put})

    @abstractmethod
    def get_price(self, stock_paths, maturity, interest_rate, option_type="C", strike_price=None):
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
        pass

    @abstractmethod
    def get_prices_paths(self, stock_paths, maturity, interest_rate, option_type="C", strike_price=None):
        raise NotImplementedError
        pass
