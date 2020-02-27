from OptionModels.Options import OptionStyle
import numpy as np
import math


class AsianMean(OptionStyle):

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
