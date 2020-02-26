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

    def get_price(self, stock_paths, maturity, interest_rate, option_type="C", strike_price=None):
        """
            A lookback option is an option where the strike_price depends on the minimum or maximum value of the path.
            Therefor the Lookback option is path-dependent.

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
        # setting the correct values for the lookback option
        stock_prices = stock_paths[:, -1]
        # if the lookback option uses the min or max
        strike_prices = np.min(stock_paths, axis=1) if self.lookback_min else np.max(stock_paths, axis=1)

        # check if the option_type is correct
        option_function = self.optiontype_dict.get(option_type)
        if option_function is None:
            raise ValueError("Invalid option_type")

        # the price under the risk natural measure
        option_price = math.e ** (-maturity * interest_rate) * np.mean(option_function(stock_prices, strike_prices))

        # Variance of the prices
        var_option_pricing = np.var(math.e ** (-maturity * interest_rate) *
                                    option_function(stock_prices, strike_prices))

        return option_price, var_option_pricing

    def get_prices_paths(self, stock_paths, maturity, interest_rate, option_type="C", strike_price=None):
        raise NotImplementedError
    # todo dit schrijven
