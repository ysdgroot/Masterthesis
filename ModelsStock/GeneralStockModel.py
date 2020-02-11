from abc import ABC, abstractmethod


class StockModel(ABC):

    @abstractmethod
    def get_stock_prices(self, amount, start_price, maturity, steps_per_maturity=100, seed=42):
        pass

    def get_price_simulations(self,
                              option_styles,
                              amount_paths,
                              start_price,
                              maturity,
                              interest_rate,
                              strike_price=None,
                              option_type='C',
                              steps_per_maturity=100,
                              seed=None):
        """
        The stock_paths will be generated of the object on which this method is called.

        :param option_styles: A list of class OptionStyle.
        :param amount_paths: Positive integer.
                            This is the total number of paths generated.
        :param start_price: Positive float.
                            Is the starting price of the stock.
        :param maturity: Positive integer.
                        The total time period for the simulation.
                        The period of one payment of the interest_rate should be the same as maturity=1.
        :param interest_rate: a positive value.
                        The interest rate per maturity.
        :param strike_price: Positive float.
                            The price for the stock when the option is exercised.
        :param option_type: 'C' for a call option, 'P' for a put option
                            For both give value ['C','P']
        :param steps_per_maturity: A positive integer. (default = 100)
                                    The amount of small steps taken to represent 1 maturity passing.
                                    The higher the number te more accurate it represents the stock,
                                        but more time consuming
        :param seed:Positive integer or None. (default = None)
                    If value is different from None, the function np.random.seed(seed) will be called.
                    For replication purposes, to get same 'random' values.

        :return: List with all the prices (positive values) of the options.
            Same length and order of the prices as the given list option_styles.
            If option_type is list (=['C','P']), then it is a dict with keys 'C' and 'P',
                with resp. the prices of the options.
        """
        # Generation of all the paths.
        simulations = self.get_stock_prices(amount_paths,
                                            start_price,
                                            maturity,
                                            steps_per_maturity=steps_per_maturity,
                                            seed=seed)

        # if it is not a list, make it a list
        if type(option_styles) is not list:
            option_styles = [option_styles]

        dict_values = {}
        if type(option_type) is list:
            for opt_type in option_type:
                dict_values[opt_type] = [option_style.get_price(simulations,
                                                                maturity,
                                                                interest_rate,
                                                                option_type=opt_type,
                                                                strike_price=strike_price)
                                         for option_style in option_styles]
            return dict_values
        else:
            # Make list with the prices for each option style
            prices = [option_style.get_price(simulations,
                                             maturity,
                                             interest_rate,
                                             option_type=option_type,
                                             strike_price=strike_price)
                      for option_style in option_styles]
            return prices
