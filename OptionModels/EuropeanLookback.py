from OptionModels.Options import OptionStyle
import numpy as np
import math


class Lookback(OptionStyle):

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
