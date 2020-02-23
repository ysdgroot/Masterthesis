from scipy.stats import norm
import numpy as np
from ModelsStock.GeneralStockModel import StockModel
import matplotlib.pyplot as plt


class BlackScholes(StockModel):

    def __init__(self, interest_rate, volatility):
        """
        :param interest_rate: Positive float.
                            The risk-free interest rate, per time maturity.
        :param volatility: Positive float.
                            The volatility of the stock, per time maturity.
        """
        self.interest_rate = interest_rate
        self.volatility = volatility

    def get_stock_prices(self, amount_paths, start_price, maturity, steps_per_maturity=100, seed=None):
        """
        Simulations of stock prices based on the Black Scholes model,
        this means the stock prices follows the Geometric Brownian Motion.

        :param amount_paths: Positive integer.
                            This is the total number of paths generated.
        :param start_price: Positive float.
                            Is the starting price of the stock.
        :param maturity: Positive integer.
                        The total time period for the simulation.
                        The period of one payment of the interest_rate should be the same as maturity=1.
        :param steps_per_maturity: A positive integer. (default = 100)
                                    The amount of small steps taken to represent 1 maturity passing.
                                    The higher the number te more accurate it represents the stock,
                                        but more time consuming
        :param seed: Positive integer or None. (default = None)
                    If value is different from None, the function np.random.seed(seed) will be called.
                    For replication purposes, to get same 'random' values.

        :return: 2d numpy.array of all the generated paths, based on the Black Scholes model.
                shape:
                        (amount, maturity * time_step_per_maturity + 1)
                Each row represents a different path, the columns the time.
                The first column is the start_price.
        """
        # set value of seed
        if seed is not None:
            np.random.seed(seed=seed)

        number_of_steps = maturity * steps_per_maturity  # the amount of timesteps needed for the stockprice
        dt = 1 / steps_per_maturity  # length for each step, to make it discrete

        # calculates a path of factors starting from start_price.
        # Each step is normal distributed with mean 0 and variance dt (length of a step)
        weiner_processes = np.cumprod(1 + self.interest_rate * dt +
                                      self.volatility * np.random.normal(0, np.sqrt(dt),
                                                                         (amount_paths, number_of_steps)),
                                      1) * start_price

        # starting prices, for the first column
        first_column = np.ones((amount_paths, 1)) * start_price

        # adding start_price as first element
        weiner_processes = np.append(first_column, weiner_processes, axis=1)

        return weiner_processes

    @staticmethod
    def help_function(start_price, strike_price, maturity, interest_rate, volatility):
        """
        Help function for the Black Scholes formula. Calculation of an important part of the formula.
        Return d_plus,d_min mostly used in the literature.

        :param maturity: Positive integer or np.array with same length as any other parameters.
                        The time that the option matures.
                        The period of one payment of the interest_rate should be the same as maturity=1.
        :param start_price: Positive float or np.array with same length as any other parameters.
                            Is the starting price of the stock.
        :param strike_price: Positive float or np.array with same length as any other parameters.
                            The price for the stock when the option is exercised.
        :param interest_rate: Positive float or np.array with same length as any other parameters.
                            The risk-free interest rate, per time maturity.
        :param volatility: Positive float or np.array with same length as any other parameters.
                            The volatility of the stock, per time maturity.
        :return: 2 values, d_plus and d_min.
            start_price = S
            strike_price = K
            interest_rate = r
            maturity = T

            d_plus//_min = (log(S/K) + (r +//- (volatility^2)/2) * T) / (volatility * sqrt(T))

        """
        d_plus = (np.log(start_price / strike_price) + (interest_rate + (volatility ** 2) / 2) * maturity) / \
                 (volatility * np.sqrt(maturity))
        d_min = d_plus - volatility * np.sqrt(maturity)

        return d_plus, d_min

    @staticmethod
    def solution_call_option(start_price, strike_price, maturity, interest_rate, volatility):
        """
            Calculating the call option based on the Black Scholes model.
            The Stock price is the Geometric Brownian Motion
            This assumes the interest rate and volatility are constant during the time of maturity.

            :param maturity: Positive integer or np.array with same length as any other parameters.
                        The time that the option matures.
                        The period of one payment of the interest_rate should be the same as maturity=1.
            :param start_price: Positive float or np.array with same length as any other parameters.
                            Is the starting price of the stock.
            :param strike_price: Positive float or np.array with same length as any other parameters.
                                The price for the stock when the option is exercised.
            :param interest_rate: Positive float or np.array with same length as any other parameters.
                                The risk-free interest rate, per time maturity.
            :param volatility: Positive float or np.array with same length as any other parameters.
                                The volatility of the stock, per time maturity.
            :return: Positive float or np.array of the same length as the input
                    Price of a call-option based on the BS model
            """
        d_plus, d_min = BlackScholes.help_function(start_price, strike_price, maturity, interest_rate, volatility)

        return start_price * norm.cdf(d_plus) - strike_price * np.exp(-interest_rate * maturity) * norm.cdf(d_min)

    @staticmethod
    def solution_put_option(start_price, strike_price, maturity, interest_rate, volatility):
        """
            Calculating the put option based on the Black Scholes model.
            The Stock price is the Geometric Brownian Motion
            This assumes the interest rate and volatility are constant during the time of maturity.

            :param maturity: Positive integer or np.array with same length as any other parameters.
                        The time that the option matures.
                        The period of one payment of the interest_rate should be the same as maturity=1.
            :param start_price: Positive float or np.array with same length as any other parameters.
                            Is the starting price of the stock.
            :param strike_price: Positive float or np.array with same length as any other parameters.
                                The price for the stock when the option is exercised.
            :param interest_rate: Positive float or np.array with same length as any other parameters.
                                The risk-free interest rate, per time maturity.
            :param volatility: Positive float or np.array with same length as any other parameters.
                                The volatility of the stock, per time maturity.
            :return: Positive float or np.array of the same length as the input
                    Price of a put-option based on the BS model.
            """
        d_plus, d_min = BlackScholes.help_function(start_price, strike_price, maturity, interest_rate, volatility)

        return strike_price * np.exp(-interest_rate * maturity) * norm.cdf(-d_min) - start_price * norm.cdf(-d_plus)

    @staticmethod
    def generate_random_variables(amount,
                                  stock_price_bound,
                                  strike_price_bound,
                                  maturity_bound,
                                  interest_rate_bound,
                                  volatility_bound,
                                  forward_pricing=False,
                                  seed=None):
        """
        Generation of random values for the Back Scholes model.

        :param amount: positive integer, number of random variables generated
        :param stock_price_bound: float, tuple or list; only the first 2 elements will be used.
                                Bounds where the values of the stock prices will be.
                                The values will be uniformly selected.
        :param strike_price_bound: float, tuple or list; only the first 2 elements will be used.
                                Bounds are the the strike prices, but are the percentages(!) of the stock_price.
                                The values will be uniformly selected.
                                (!) If forward_pricing = True, then the strike_prices are the percentage
                                    of the forward pricing (=e^(r*T)*S0)
        :param maturity_bound: int, tuple or list; only the first 2 elements will be used.
                                Bounds where the values of the maturity will be.
                                The values will be uniformly selected.
        :param interest_rate_bound: float, tuple or list; only the first 2 elements will be used.
                                Bounds where the values of the interest rate will be.
                                The values will be uniformly selected.
        :param volatility_bound: float, tuple or list; only the first 2 elements will be used.
                                Bounds where the values of the volatility will be.
                                The values will be uniformly selected.
        :param forward_pricing: Boolean (default = False)
                                If the strike_prices are the percentage of the forward pricing
                                    instead of the (start) stock prices.
        :param seed: Positive integer or None. (default = None)
                    If value is different from None, the function np.random.seed(seed) will be called.
                    For replication purposes, to get same 'random' values.
        :return: dict, with keys: "interest_rate", "volatility", "maturity", "stock_price", "strike_price",
                                "strike_price_percent", "forward_pricing"
                For each key the values are a np.array of length 'amount' with the random values,
                    but "forward_pricing" is True or False
                        if the percentage of the forward pricing as strike price has been used or not.
        """

        def conversion_and_check(value):
            # convert value into a tuple in increasing order.
            # control if the values are positive
            if len(value) == 1:
                bounds = (value, value)
            elif len(value) >= 2:
                lower = min(value[0], value[1])
                upper = max(value[0], value[1])
                bounds = (lower, upper)
            else:
                raise TypeError

            # only 1 check is necessary, because this is the minimum.
            if bounds[0] < 0:
                raise ValueError
            return bounds

        # set seed
        if seed is not None:
            np.random.seed(seed=seed)

        # conversion to a tuple, in increasing order and controls if the values are positive.
        stock_price_bound = conversion_and_check(stock_price_bound)
        strike_price_bound = conversion_and_check(strike_price_bound)
        maturity_bound = conversion_and_check(maturity_bound)
        interest_rate_bound = conversion_and_check(interest_rate_bound)
        volatility_bound = conversion_and_check(volatility_bound)

        # Check if the maturity is an integer.
        if type(maturity_bound[0]) is not int or type(maturity_bound[1]) is not int:
            raise ValueError

        # random Integer selection for the maturities
        maturities = np.random.randint(maturity_bound[0], maturity_bound[1], amount)

        # random data selection, Uniform
        interest_rates = np.random.uniform(interest_rate_bound[0], interest_rate_bound[1], amount)
        volatilities = np.random.uniform(volatility_bound[0], volatility_bound[1], amount)
        stock_prices = np.random.uniform(stock_price_bound[0], stock_price_bound[1], amount)
        strike_prices_percentage = np.random.uniform(strike_price_bound[0], strike_price_bound[1], amount)

        # Take a percentage of the stock price.
        # If forward pricing, the strike price is the percentage of the forward price.
        strike_prices = stock_prices * strike_prices_percentage if not forward_pricing \
            else stock_prices * np.exp(interest_rates * maturities) * strike_prices_percentage

        # todo test of het weldegelijk forward pricing neemt als het True/False is

        # making dictionary for each parameter
        data_dict = {"interest_rate": interest_rates,
                     "volatility": volatilities,
                     "maturity": maturities,
                     "stock_price": stock_prices,
                     "strike_price": strike_prices,
                     "strike_price_percent": strike_prices_percentage,
                     "forward_pricing": forward_pricing}

        return data_dict


########################################################################################################################
# import numpy.random as nprand
# import pandas as pd
#
# def get_random_data(amount, price_bound, interest_rate_bound, volatility_bound, maturity_bound, strike_price_bound,
#                     seed=113):
#     """
#
#     :param amount:
#     :param price_bound:
#     :param interest_rate_bound:
#     :param volatility_bound:
#     :param maturity_bound:
#     :param strike_price_bound:
#     :param seed:
#     :return:
#     """
#
#     nprand.seed(seed)
#
#     def data_changer(obj):
#         """
#         Helpfunction to check if an obj has at least 2 elements (length != 1).
#         If not it will return the object twice
#         :param obj:
#         :return:
#         """
#         value = obj
#         if isinstance(obj, (int, float)) and not isinstance(obj, bool):
#             value = (obj, obj)
#         return value
#
#     # Make tuples of the data, if they are just values
#     price_bound = data_changer(price_bound)
#     interest_rate_bound = data_changer(interest_rate_bound)
#     volatility_bound = data_changer(volatility_bound)
#     maturity_bound = data_changer(maturity_bound)
#     strike_price_bound = data_changer(strike_price_bound)
#
#     # Generate with the bounds random values that are uniformly distributed
#     prices = nprand.uniform(price_bound[0], price_bound[1], amount)
#     interest_rates = nprand.uniform(interest_rate_bound[0], interest_rate_bound[1], amount)
#     volatilities = nprand.uniform(volatility_bound[0], volatility_bound[1], amount)
#     maturities = nprand.uniform(maturity_bound[0], maturity_bound[1], amount)
#     strike_prices = nprand.uniform(strike_price_bound[0], strike_price_bound[1], amount)
#
#     data_dict = {'Price': prices, 'Strike_price': strike_prices, 'Interest_rate': interest_rates,
#             'Volatility': volatilities, 'Maturity': maturities}
#     data = pd.DataFrame(data=data_dict)
#
#     # prices, interest_rates, volatilities, maturities, strike_prices
#     return data
#
#
# def get_random_solutions(option, prices, interest_rates, volatilities, maturities, strike_prices):
#     if option == 'C':
#         solution = BlackScholes.solution_call_option(prices, strike_prices, maturities, interest_rates, volatilities)
#     else:
#         solution = BlackScholes.solution_put_option(prices, strike_prices, maturities, interest_rates, volatilities)
#
#     return solution
#
# def get_random_data_and_solutions(option, amount, price_bound, interest_rate_bound, volatility_bound, maturity_bound,
#                                   strike_price_bound, seed=113):
#     """
#
#     :param option:
#     :param amount:
#     :param price_bound:
#     :param interest_rate_bound:
#     :param volatility_bound:
#     :param maturity_bound:
#     :param strike_price_bound:
#     :param seed:
#     :return:
#     """
#
#
#     data = get_random_data(amount, price_bound, interest_rate_bound, volatility_bound, maturity_bound,
#                            strike_price_bound, seed=seed)
#
#     prices = data['Price']
#     interest_rates = data['Interest_rate']
#     volatilities = data['Volatility']
#     maturities = data['Maturity']
#     strike_prices = data['Strike_price']
#
#     results = get_random_solutions(option, prices, interest_rates, volatilities, maturities, strike_prices)
#     data['Value_option'] = results
#
#     return data
#
