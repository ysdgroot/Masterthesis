from OptionModels.Options import OptionStyle
import numpy as np
import math


class AsianMean(OptionStyle):

    # todo: verwijderen van dit algemeen geval, te weinig tijd om dit effectief te doen.
    def __init__(self, period_mean=None, steps_per_maturity=100):
        # todo: documentatie
        super(AsianMean, self).__init__()
        self.period_mean = period_mean
        self.time_step_per_maturity = steps_per_maturity

        # search the number of columns back in the matrix of stock_paths
        if period_mean is None:
            self.steps_back = None
        else:
            self.steps_back = period_mean * steps_per_maturity

    def get_price(self, stock_paths, maturity, interest_rate, option_type="C", strike_price=None):
        """
        Calculates the price of a European option with Asian pricing, given all the stock_paths.
        The strike_price is calculated as the mean of the stock (stock_path).
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
        #todo: extra controles en functionaliteit met steps back (+ controle matrix groot genoeg)

        # The strike_prices by the mean of all the paths
        strike_prices = np.mean(stock_paths, axis=1)

        # The values of each stock
        prices_stock = stock_paths[:, -1]

        # check if the option_type is correct
        if option_type in self.optiontype_dict.keys():
            # Get the function of the put or call option
            # option_function = self.get_dict()[option_type]
            option_function = self.optiontype_dict[option_type]
        else:
            raise ValueError("Invalid option_type")

        # the price under the risk natural measure
        return math.e ** (-maturity * interest_rate) * np.mean(option_function(prices_stock, strike_prices))
