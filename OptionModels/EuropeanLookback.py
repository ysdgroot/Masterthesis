from OptionModels.Options import OptionStyle
import numpy as np
import math

class Lookback(OptionStyle):

    def __init__(self, lookback_min=True, floating_lookback=True):
        #todo: documentatie
        super(Lookback, self).__init__()
        self.lookback_min = lookback_min
        self.floating_lookback = floating_lookback

    def get_price(self, stock_paths, maturity, interest_rate, option_type="C", strike_price=None):
        """
            A Floating strike lookback option (see https://www.investopedia.com/terms/l/lookbackoption.asp)
                is an option where the strike_price depends on the minimum or maximum value of the path.
            Therefor the Lookback option is path-dependent.
            A Fixed strike lookback option (floating=False) is an option where the strike_price is a fixed value,
                but the price of the stock is the minimum or maximum value.

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
            :param strike_price: a positive value or None. (default=None)
                            This value will be ignored when using a floating_lookback (=True, default value)
                            Otherwise (floating_lookback=False) the strike_price must be different from None

            :return: A positive value, which represents the price of the option.
            """
        # TODO: schrijven van documentie (niet vergeten floating vs fixed lookback)

        # if the lookback option uses the min or max
        values = np.min(stock_paths, axis=1) if self.lookback_min else np.max(stock_paths, axis=1)

        # setting the correct values for a floating or fixed lookback option
        if self.floating_lookback:
            prices_stock = stock_paths[:, -1]
            strike_prices = values
        else:
            prices_stock = values
            strike_prices = strike_price

        # check if the option_type is correct
        if option_type in self.optiontype_dict.keys():
            # Get the function of the put or call option
            # option_function = self.get_dict()[option_type]
            option_function = self.optiontype_dict[option_type]
        else:
            raise ValueError("Invalid option_type")

        # the price under the risk natural measure
        return math.e ** (-maturity * interest_rate) * np.mean(option_function(prices_stock, strike_prices))
