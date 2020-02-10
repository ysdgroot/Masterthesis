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

    #
    # def get_price_simulation(self, stock_model, amount_paths, start_price, maturity, strike_price=None,
    #                              option_type="C", time_step_per_maturity=100, seed=42):
    #     """
    #     Get the price of the option based on a stock_model, which simulates amount_paths different paths for the stock.
    #     The simulation starts with the start_price and maturity.
    #     Because this is a simulation, the path makes some fixed small steps,
    #         which can be altered by time_step_per_maturity. The bigger this number the better it will approximate
    #         the theoretical value.
    #     The price of the option is also based on the strike_price, default=None because some options are path-dependent,
    #         which means the strike_price depends on previous values of the stock.
    #     The call option will be set as a default (option_type), 'P' is for the put option.
    #     For random generations, the seed is set to 113, which has no meaning.
    #     For additional values for the simulation of the option, this can also be given.
    #
    #     :param stock_model: object of type 'StockModel', for the simulation of the stock.
    #     :param amount_paths: the amount (positive integer) of paths that need to be generated for the simulation.
    #     :param start_price: a number that represents the startprice of the stock
    #     :param maturity: a positive value which represents the time left of the option.
    #     :param strike_price: (default=None) if the strike_price is path dependent. Else (not-path-dependent)
    #         a value.
    #     :param option_type: (default='C') 'C' for a call option and 'P' for a put option.
    #     :param time_step_per_maturity: (default=100) a positive value.
    #     :param seed: (default=42) to redo the simulation if necessary
    #     :return: a value which represents the value of the option based on the simulations.
    #     """
    #
    #     # Get all the simulations/stockpaths based on the given StockModel
    #     simulations = stock_model.get_stock_prices(amount_paths, start_price, maturity,
    #                                                time_step_per_maturity=time_step_per_maturity, seed=seed)
    #
    #
    #     return self.get_price(simulations, option_type=option_type, strike_price=strike_price)

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

