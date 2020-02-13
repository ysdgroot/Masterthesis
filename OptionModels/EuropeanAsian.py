from OptionModels.Options import OptionStyle
import numpy as np
import math


class AsianMean(OptionStyle):

    def __init__(self, period_mean=None):
        """
        Object to represent the Asian option of European style.

        For Asian options the strike price is based on the mean value of the stock.
        Period_mean is to define the amount of periods the strike price is based on.
        :param period_mean: an positive integer,
            For the amount of periods it needs to go back to calculate the strike price of this option.
        """
        super(AsianMean, self).__init__()
        if period_mean is not None and (type(period_mean) is not int or period_mean < 0):
            raise TypeError("Invalid type given, must be a positive integer")
        self.period_mean = period_mean

    def get_price(self, stock_paths, maturity, interest_rate, option_type="C", strike_price=None):
        """
        Calculates the price of a European option with Asian pricing, given all the stock_paths.
        The strike_price is calculated as the mean of the stock (stock_paths),
                with the assumption that the first column is the starting price of the stock (all the same)
        Therefor the strike_price is path-dependent.

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
                            This value will be ignored, because the strike_price is based on the path.

        :return: A positive value, which represents the price of the option.
        """

        if self.period_mean is not None:
            # Works correctly.
            # But if you want the price 'at exactly' the time before period_mean you need
            #                                                               to substract -1 at position_steps_back.
            # unless the first column is not the starting price.
            time_steps_per_maturity = stock_paths.shape[1] // maturity
            position_steps_back = stock_paths.shape[1] - time_steps_per_maturity * min(self.period_mean, maturity)
            strike_prices = np.mean(stock_paths[:, position_steps_back:], axis=1)
        else:
            # The strike_prices by the mean of all the paths
            strike_prices = np.mean(stock_paths, axis=1)

        # The values of each stock
        prices_stock = stock_paths[:, -1]

        # check if the option_type is correct
        option_function = self.optiontype_dict.get(option_type)
        if option_function is None:
            raise ValueError("Invalid option_type")

        # the price under the risk natural measure
        return math.e ** (-maturity * interest_rate) * np.mean(option_function(prices_stock, strike_prices))
