from abc import ABC, abstractmethod
import numpy as np
from OptionModels import Options


class StockModel(ABC):

    @abstractmethod
    def get_stock_prices(self, amount, start_price, maturity, steps_per_maturity=100, seed=None):
        pass

    def get_price_simulations(self,
                              option_styles: Options,
                              amount_paths,
                              start_price,
                              maturity,
                              interest_rate,
                              strike_price=None,
                              option_type='C',
                              steps_per_maturity=100,
                              seed=None,
                              max_path_generation=None):
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
        :param max_path_generation: positive int (not 0) or None. (default = None)
                    It is the maximum number of paths that are generated for the option pricing.
                    This is to reduce unnecessary RAM memory, but slower in speed.
                    If value is None, all the paths will be generated at the same time.

        :return: List with all the prices (positive values) of the options.
            Same length and order of the prices as the given list option_styles.
            If option_type is list (=['C','P']), then it is a dict with keys 'C' and 'P',
                with resp. the prices of the options.
        """

        def partition_maker(total_number, value_splitter):
            deler, rest = divmod(total_number, value_splitter)

            values = [value_splitter] * deler
            if rest != 0:
                values += [rest]

            return values

        if type(max_path_generation) is not int:
            raise TypeError("max_path_generation must be an integer")

        value_splitter = amount_paths if max_path_generation is None else max_path_generation
        number_paths_generating = partition_maker(amount_paths, value_splitter)

        if seed is not None:
            np.random.seed(seed)

        # if it is not a list, make it a list
        if type(option_styles) is not list:
            option_styles = [option_styles]

        names_options = [str(option) for option in option_styles]

        # dict if option_type is a list
        dict_values_list = {}
        # dict if option_type is 'C' or 'P'
        dict_paths = {}

        # starting to generate a smaller amount of paths, to use less RAM memory
        for n_paths in number_paths_generating:
            # Generation of all the paths.
            simulations = self.get_stock_prices(n_paths,
                                                start_price,
                                                maturity,
                                                steps_per_maturity=steps_per_maturity)

            for option_style in option_styles:
                if type(option_type) is list:
                    for opt_type in option_type:
                        correct_dict_paths = dict_values_list.get(opt_type, dict())

                        prices_option_paths = correct_dict_paths.get(str(option_style), [])
                        new_prices = option_style.get_prices_per_path(simulations,
                                                                      maturity,
                                                                      interest_rate,
                                                                      option_type=opt_type,
                                                                      strike_price=strike_price)
                        prices_option_paths.extend(new_prices)
                        correct_dict_paths[str(option_style)] = prices_option_paths

                        dict_values_list[opt_type] = correct_dict_paths
                else:
                    prices_option_paths = dict_paths.get(str(option_style), [])
                    new_prices = option_style.get_prices_per_path(simulations,
                                                                  maturity,
                                                                  interest_rate,
                                                                  option_type=option_type,
                                                                  strike_price=strike_price)
                    prices_option_paths.extend(new_prices)
                    dict_paths[str(option_style)] = prices_option_paths

        # set everything in the correct structure to return
        if type(option_type) is list:
            for opt_type in option_type:
                prices = [np.mean(dict_values_list.get(opt_type)[name]) for name in names_options]
                dict_values_list[opt_type] = prices
            return dict_values_list
        else:
            prices = [np.mean(dict_paths[name]) for name in names_options]
            return prices
        # dict_values = {}
        # if type(option_type) is list:
        #     # If the option_type is a list, we make a dictionary to get the correct values.
        #     for opt_type in option_type:
        #         dict_values[opt_type] = [option_style.get_price_option(simulations,
        #                                                                maturity,
        #                                                                interest_rate,
        #                                                                option_type=opt_type,
        #                                                                strike_price=strike_price)[0]
        #                                  for option_style in option_styles]
        #     return dict_values
        # else:
        #     # Make list with the prices for each option style
        #     prices = [option_style.get_price_option(simulations,
        #                                             maturity,
        #                                             interest_rate,
        #                                             option_type=option_type,
        #                                             strike_price=strike_price)[0]
        #               for option_style in option_styles]
        #     return prices
