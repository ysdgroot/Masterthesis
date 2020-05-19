import pandas as pd
import importlib.resources as pkg_resources
import GeneratedData
import Data


class DataManager:

    def __init__(self, stockmodel="BS",
                 option_type="opt_standard",
                 replace_call_put=True,
                 call_put=(1, -1),
                 only_call=False,
                 list_column_names=None,
                 shuffle_data=False,
                 with_percent=False):
        """
        :param stockmodel:str (default="BS", values in ["BS", "VG", "H"]
        :param option_type: str (default="opt_standard"), standard values are
                                                ["opt_standard",
                                                "opt_asianmean",
                                                "opt_lookbackmin",
                                                "opt_lookbackmax"]
                                    for the BS stockmodel, there is the possibility of "opt_exact_standard"
        :param replace_call_put: bool (default=True), if the column "call/put" needs to be changed
        :param call_put: 2-tuple (default=(1,-1)), when replace_call_put=True, then it will be changed in these values.
                                                first value is the 'call', second the 'put'
        :param only_call: bool (default=False), if you only need the call_options
        :param list_column_names: list[str] (default=None), if a specific column set is necessary as training data.
                                If None, all the columns will be given.
        :param shuffle_data: bool (default=False), whenever the data needs to be shuffled.
        """
        self.replace_call_put = replace_call_put
        self.call_put = call_put

        self.possible_models = ["BS", "VG", "H"]

        self.standard_columns_x = ["stock_price",
                                   "strike_price",
                                   "strike_price_percent",
                                   "interest_rate",
                                   "maturity",
                                   "call/put"] if with_percent else ["stock_price",
                                                                     "strike_price",
                                                                     "interest_rate",
                                                                     "maturity",
                                                                     "call/put"]
        self.standard_columns_y = ["opt_standard",
                                   "opt_asianmean",
                                   "opt_lookbackmin",
                                   "opt_lookbackmax"]

        self.dict_additional_col_x = {"BS": ["volatility"],
                                      "VG": ["skewness", "volatility", "kurtosis"],
                                      "H": ["start_vol",
                                            "long_term_vol",
                                            "rate_reversion",
                                            "vol_of_vol",
                                            "correlation"]}
        self.dict_additional_col_y = {"BS": ["opt_exact_standard"],
                                      "VG": [],
                                      "H": []}

        if not stockmodel in self.possible_models:
            raise ValueError(f'{stockmodel} is not a possible modelname')
        self.model = stockmodel

        self.column_fitting = option_type
        self.set_column_fitting(option_type)

        self.list_column_names = None
        self.set_column_names(list_column_names=list_column_names)

        # get filenames
        self.training_file_name = self.get_correct_file(self.model, test=False)
        self.test_file_name = self.get_correct_file(self.model, test=True)

        self.only_call = only_call
        self.shuffle_data = shuffle_data

    def func_replace_call_put(self, df, call_put=None):
        """
        Replace the values of the call/put column, to the values of self.call_put
        :param df: pandas.dataframe, with column "call/put"
        :param call_put: None or 2-tuple (default=None), the values in which they need to change.
                        When None, the values are the values from self.call_put
        :return: None
        """

        if call_put is None:
            call_put = self.call_put

        if self.replace_call_put:
            df.loc[df["call/put"] == 'C', "call/put"] = call_put[0]
            df.loc[df["call/put"] == 'P', "call/put"] = call_put[1]

    @staticmethod
    def get_correct_file(model, test=False):
        """
        Get the correct file_name where the data is stored.

        :param model: str, to get the correct file_name
        :param test: bool (default: False), if the file is test data or not
        :return: full path name of the file
        """
        try:
            file = pkg_resources.open_text(GeneratedData,
                                           f"generateddata-{model}-"
                                           f"{'testing' if test else 'training'}.csv")
            # data_folder = Path('C:/Users/Yucatan De Groote/Documents/Universiteit/Masterproef/GeneratedData')
            return file.name
        except FileNotFoundError:
            print("File not found: Incorrect stockmodel name")
            return None

    def get_only_call(self, only_call=True):
        """
        Function to get only the call values

        :param only_call: bool, whenever to get only the call values.
        :return: returns an instance of self.
        """
        self.only_call = only_call
        return self

    def set_column_fitting(self, column_fitting):
        """
        Function to change the value of the column of the y-value

        :param column_fitting: str, the column name as y value
        :return: returns an instance of self.
        """
        if not column_fitting in self.standard_columns_y + self.dict_additional_col_y[self.model]:
            raise ValueError(f"{column_fitting} is not a column that exists")
        self.column_fitting = column_fitting
        return self

    def with_percent(self):
        self.set_column_fitting(self.standard_columns_x)

    def set_column_names(self, list_column_names):
        """
        Set the column names for the Training and Test data.
        This will be a check if all the columns names are correct

        :param list_column_names: list[str], list with all the column names
        :return: returns an instance of self.
        """

        if list_column_names is not None:
            temp_list_columns_x = self.standard_columns_x + self.dict_additional_col_x[self.model]
            if not all(column in temp_list_columns_x for column in list_column_names):
                raise ValueError(f"The list of column names contains elements which are not possible")
        self.list_column_names = list_column_names
        return self

    def get_training_data(self):
        """
        Function to get the training data from the csv-file

        :return: (X, y), pandas.dataframe,
                        X: dataframe with all the columns to train the models (or only the list_column_names)
                        y: dataframe (just one column) with the results of the X values.
        """
        # read the csv files
        training_data = pd.read_csv(self.training_file_name, comment='#', header=0)

        if self.shuffle_data:
            training_data = training_data.sample(frac=1).reset_index(drop=True)

        # replacing the values for call-put
        self.func_replace_call_put(training_data)

        columns_x = self.standard_columns_x + self.dict_additional_col_x[self.model]
        columns_y = self.column_fitting

        # getting the positions in case only the calls are asked
        positions_call = training_data["call/put"] == self.call_put[0]

        training_data_x = training_data[columns_x] if not self.only_call \
            else training_data[columns_x][positions_call]

        training_data_y = training_data[columns_y] if not self.only_call \
            else training_data[columns_y][positions_call]

        if self.list_column_names is not None:
            training_data_x = pd.DataFrame(training_data_x, columns=self.list_column_names)

        return training_data_x, training_data_y

    def get_random_training_data(self, n_samples=10000, random_state=8843, get_not_selected_data=False):
        """
        Function to get random samples from the full dataset.
        :param n_samples: int, number of datapoints
        :param random_state: int, for the randomstate.
        :param get_not_selected_data: bool (default=False), to get the data which are not from the random selected values.
        :return: (X, y), pandas.dataframe,
                        X: dataframe with all the columns to train the models (or only the list_column_names)
                        y: dataframe (just one column) with the results of the X values.
        """
        training_data = pd.read_csv(self.training_file_name, comment='#', header=0)

        # replacing the values for call-put
        self.func_replace_call_put(training_data)

        columns_x = self.standard_columns_x + self.dict_additional_col_x[self.model]
        columns_y = self.column_fitting

        # getting the positions in case only the calls are asked
        positions_call = training_data["call/put"] == self.call_put[0]

        # select only the call options
        if self.only_call:
            training_data = training_data[positions_call]

        training_data_sample = training_data.sample(n=n_samples, random_state=random_state)
        training_data_sample_index = training_data_sample.index

        if self.list_column_names is not None:
            training_data_x = pd.DataFrame(training_data_sample, columns=self.list_column_names)
        else:
            training_data_x = training_data_sample[columns_x]

        training_data_y = training_data_sample[columns_y]

        if get_not_selected_data:
            not_selected_x = training_data.loc[set(training_data.index) - set(training_data_sample_index)][columns_x]
            not_selected_y = training_data.loc[set(training_data.index) - set(training_data_sample_index)][columns_y]

            return training_data_x, training_data_y, not_selected_x, not_selected_y
        else:
            return training_data_x, training_data_y

    def get_test_data(self):
        """
        Function to get the test data from the csv-file

        :return: (X, y), pandas.dataframe,
                        X: dataframe with all the columns to test the models (or only the list_column_names)
                        y: dataframe (just one column) with the results of the X values.
        """
        # read the csv files
        test_data = pd.read_csv(self.test_file_name, comment='#', header=0)

        # replacing the values for call-put
        self.func_replace_call_put(test_data)

        columns_x = self.standard_columns_x + self.dict_additional_col_x[self.model]
        columns_y = self.column_fitting

        positions_call = test_data["call/put"] == self.call_put[0]

        test_data_x = test_data[columns_x] if not self.only_call else test_data[columns_x][positions_call]

        test_data_y = test_data[columns_y] if not self.only_call else test_data[columns_y][positions_call]

        if self.list_column_names is not None:
            test_data_x = pd.DataFrame(test_data_x, columns=self.list_column_names)

        return test_data_x, test_data_y

    def get_full_dataframe(self, test_data=False, change_call_put=False, call_put=None):
        """
        Function to get the full dataframe of the training or test file.
        :param test_data: boolean (default=False), whenever to get the test(=True) data of the training(=False) data
        :return: pandas.Dataframe with the training/test data, with all the columns of the stockmodel from the declaration
        """
        file_name = self.test_file_name if test_data else self.training_file_name
        data = pd.read_csv(file_name, comment='#', header=0)

        if change_call_put:
            self.func_replace_call_put(data, call_put=call_put)

        return data


class DataManagerRealData:

    def __init__(self, symbol="SPX", test_month=9):
        """
        :param symbol: str, ("SPX", "SPXPM" or "SX5E) (default="SPX")
        :param test_month: the month used as test data
        """
        self.symbol = symbol
        self.test_month = test_month
        self.columns_x = ["strike",
                          "call_put",
                          "stock_price_for_iv",
                          "number_days"]
        self.columns_y = ["mean_price"]

    def get_file_name(self):
        """
        :return: str or None, The full name (adress) of the file
                if None the file is not found,
        """
        symbol = self.symbol
        try:
            # file = pkg_resources.open_text(Data, f"Real data {symbol}.csv")
            file = pkg_resources.open_text(Data, f"{symbol}.csv")
            return file.name
        except FileNotFoundError:
            print("File not found")
            return None

    def get_filter(self):
        """
        Function to get the start period and end period of the month (self.test_month)

        :return:(begin, end), both Timestamps of the test month
        """
        first_check = pd.Timestamp(year=2014, month=self.test_month, day=1)

        if self.test_month != 12:
            second_check = pd.Timestamp(year=2014, month=self.test_month + 1, day=1)
        else:
            second_check = pd.Timestamp(year=2015, month=1, day=1)

        return first_check, second_check

    def get_training_data(self):
        """
        :return: (X, y) of the training data.
            All the datapoints but a month (`test_month')
        """

        df = pd.read_csv(self.get_file_name(), header=0, parse_dates=["date"])

        first_check, second_check = self.get_filter()

        filter = (df["date"] >= first_check) & (df["date"] < second_check)

        return df[self.columns_x][-filter], df[self.columns_y][-filter]

    def get_test_data(self):
        """
        :return: (X, y) of the test data.
            The test data is a specific month
        """

        df = pd.read_csv(self.get_file_name(), header=0, parse_dates=["date"])

        first_check, second_check = self.get_filter()

        filter = (df["date"] >= first_check) & (df["date"] < second_check)

        return df[self.columns_x][filter], df[self.columns_y][filter]

    def get_random_training_test_data(self, n_samples, random_state=3477):
        """
        Takes random samples of the training and test data
        :param n_samples: int, amount of datapoints for the training data
        :return: (X_train, y_train, X_test, y_test)
        """

        df = pd.read_csv(self.get_file_name(), header=0, parse_dates=["date"])

        training_data = df.sample(n_samples, random_state=random_state)

        training_data_sample_index = training_data.index

        test_data = df.loc[set(df.index) - set(training_data_sample_index)]

        return training_data[self.columns_x], training_data[self.columns_y], test_data[self.columns_x], test_data[
            self.columns_y]

    # todo: verwijderen
