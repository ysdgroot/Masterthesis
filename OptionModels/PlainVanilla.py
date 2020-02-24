from OptionModels.Options import OptionStyle
import numpy as np
import math


class PlainVanilla(OptionStyle):

    def __init__(self, strike_price_in_percent=False):
        """
        :param strike_price_in_percent: True of False. (default=False)
                                        To represent the strike_price as a percentage of the start_price.
                                        True: if strike_price needs to be used as a percentage
                                        False: other
        """
        super(PlainVanilla, self).__init__()
        self.strike_price_in_percent = strike_price_in_percent

    def get_price(self, stock_paths, maturity, interest_rate, option_type="C", strike_price=None):
        """
        Calculates the price of a plain vanilla option (standard European option), given all the stock_paths.

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
                            If None, then the strike price is the start_price (first element of the stock_paths).
                            If strike_price_in_percent = True
                                the (real) strike_price will the strike_price * start_price

        :return: A positive value, which represents the price of the option at the moment (risk natural measure)
        """

        # This check need to be before 'Strike_price=None', otherwise there can be bugs when giving 'strike_price=None'
        # and 'strike_price_in_percent=True'
        if strike_price is not None and self.strike_price_in_percent:
            strike_price = stock_paths[0, 0] * strike_price

        # Check if the strike_price is None, if so, we take the first element as strike_price
        if strike_price is None:
            strike_price = stock_paths[0, 0]

        # check if the option_type is correct
        option_function = self.optiontype_dict.get(option_type)
        if option_function is None:
            raise ValueError("Invalid option_type")

        # Gets the last value of the stock, for European vanilla option (the last column)
        stock_prices = stock_paths[:, -1]

        # the price under the risk natural measure
        option_price = math.e ** (-maturity * interest_rate) * np.mean(option_function(stock_prices, strike_price))
        # todo geef ook de variantie van de optie prijzen.
        # Variance of the prices
        var_option_pricing = np.var(math.e ** (-maturity * interest_rate) *
                                    option_function(stock_prices, strike_price))
        return option_price, var_option_pricing
