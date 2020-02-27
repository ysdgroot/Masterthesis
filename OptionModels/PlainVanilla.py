from OptionModels.Options import OptionStyle
import numpy as np
import math


class PlainVanilla(OptionStyle):

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
