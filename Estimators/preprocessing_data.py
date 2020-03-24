import pandas as pd
import importlib.resources as pkg_resources
import GeneratedData


class DataManager:

    def __init__(self, model="BS",
                 column_fitting="opt_standard",
                 replace_call_put=True,
                 call_put=(1, -1),
                 only_call=False,
                 list_column_names=None):
        """
        :param model:str (default="BS", values in ["BS", "VG", "H"]
        :param column_fitting: str (default="opt_standard"), standard values are
                                                ["opt_standard",
                                                "opt_asianmean",
                                                "opt_lookbackmin",
                                                "opt_lookbackmax"]
                                    for the BS model, there is the possibility of "opt_exact_standard"
        :param replace_call_put: bool (default=True), if the column "call/put" needs to be changed
        :param call_put: 2-tuple (default=(1,-1)), when replace_call_put=True, then it will be changed in these values.
                                                first value is the 'call', second the 'put'
        :param only_call: bool (default=False), if you only need the call_options
        :param list_column_names: list[str] (default=None), if a specific column set is necessary as training data.
                                If None, all the columns will be given.
        """
        self.replace_call_put = replace_call_put
        self.call_put = call_put

        self.possible_models = ["BS", "VG", "H"]

        self.standard_columns_x = ["stock_price",
                                   "strike_price",
                                   "strike_price_percent",
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

        if not model in self.possible_models:
            raise ValueError(f'{model} is not a possible modelname')
        self.model = model

        self.column_fitting = column_fitting
        self.set_column_fitting(column_fitting)

        self.list_column_names = None
        self.set_column_names(list_column_names=list_column_names)

        # get filenames
        self.training_file_name = self.get_correct_file(self.model, test=False)
        self.test_file_name = self.get_correct_file(self.model, test=True)

        self.only_call = only_call

    def func_replace_call_put(self, df):
        """
        Replace the values of the call/put column, to the values of self.call_put
        :param df: pandas.dataframe, with column "call/put"
        :return: None
        """
        if self.replace_call_put:
            df.loc[df["call/put"] == 'C', "call/put"] = self.call_put[0]
            df.loc[df["call/put"] == 'P', "call/put"] = self.call_put[1]

    @staticmethod
    def get_correct_file(model, test=False):
        """
        Get the correct file_name where the data is stored.

        :param model: str, to get the correct file_name
        :param test: bool (default: False), if the file is test data or not
        :return: full path name of the file
        """
        # todo: dit veranderen zodat het algemeen wordt + de namen van de files veranderen
        # f"Gen_{'test_' if test else ''}data-{model}{'' if test else '-50k'}.csv"
        try:
            file = pkg_resources.open_text(GeneratedData,
                                           f"Generated Data - {model} model -{'Test data' if test else '50k'}.csv")
            # data_folder = Path('C:/Users/Yucatan De Groote/Documents/Universiteit/Masterproef/GeneratedData')
            return file.name
        except FileNotFoundError:
            print("File not found: Incorrect model name")
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

        # replacing the values for call-put
        self.func_replace_call_put(training_data)

        columns_x = self.standard_columns_x + self.dict_additional_col_x[self.model]
        columns_y = self.column_fitting

        positions_call = training_data["call/put"] == self.call_put[0]

        training_data_x = training_data[columns_x] if not self.only_call \
            else training_data[columns_x][positions_call]

        training_data_y = training_data[columns_y] if not self.only_call \
            else training_data[columns_y][positions_call]

        if self.list_column_names is not None:
            training_data_x = pd.DataFrame(training_data_x, columns=self.list_column_names)

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

    def test_funct(self):
        print('BAL')
