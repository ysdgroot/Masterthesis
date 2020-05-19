from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from Estimators import DataCollector as dc
import matplotlib.pyplot as plt
import modelsaver
from sklearn import preprocessing, tree

########################################################
# ---------------- PARAMETERS -------------------------#
########################################################
models = ["BS", "VG", "H"]
columns_fitting = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

dict_column_to_option = {"opt_standard": "Standard",
                         "opt_asianmean": "Asian",
                         "opt_lookbackmin": "Lookback (min)",
                         "opt_lookbackmax": "Lookback (max)",
                         "opt_exact_standard": "Standard(theory)"}


########################################################################################################################


def rf_n_estimators(stockmodel="BS",
                    option_type="opt_exact_standard",
                    range_n_estimators=range(50, 751, 50),
                    save_mse=True,
                    max_features="auto",
                    scale=True):
    """
    Method to calculate the mse for a range of estimators

    :param stockmodel: str, "BS", "VG" or "H"
    :param option_type: str, "opt_standard", "opt_asianmean", "opt_lookbackmin" or "opt_lookbackmax"
            If stockmodel = "BS" -> "opt_exact_standard" is also possible
    :param range_n_estimators: list with the number of estimators for each run
    :param save_mse: bool, whenever to save all the values in a file.
    :param max_features: "auto", "log2" or a integer, for the splits in the Tree stockmodel
    :param scale: bool, if the data needs to be scaled or not
    :return: dict,with keys "Train", "Test", "oob_score", "n_estimators".
            Train = mse of the Training data
            Test = mse of the Test data
            oob_score = mse of the out-of-bag observations
            n_estimators = list of the number of estimators
    """

    dict_option_types = {"opt_exact_standard": "SE",
                         "opt_standard": "S",
                         "opt_asianmean": "A",
                         "opt_lookbackmin": "Lmin",
                         "opt_lookbackmax": "Lmax"}

    list_results_train = []
    list_results_test = []
    list_oob_score = []

    datamanager = dc.DataManager(stockmodel=stockmodel,
                                 option_type=option_type)

    X, y = datamanager.get_training_data()
    X_test, y_test = datamanager.get_test_data()

    if scale:
        scaler = preprocessing.StandardScaler().fit(X, y)
        X = scaler.transform(X)
        X_test = scaler.transform(X_test)

    for n_estimator in range_n_estimators:
        rf_model = RandomForestRegressor(n_estimators=n_estimator,
                                         verbose=1,
                                         n_jobs=7,
                                         random_state=2458 + n_estimator,
                                         max_features=max_features,
                                         oob_score=True)
        rf_model.fit(X, y)

        mse_train = mean_squared_error(y, rf_model.predict(X))
        mse_test = mean_squared_error(y_test, rf_model.predict(X_test))
        oob_score = rf_model.oob_score_

        print(f'Train {mse_train}')
        print(f'Test {mse_test}')
        print(f'OOB score: {oob_score}')

        list_results_train.append(mse_train)
        list_results_test.append(mse_test)
        list_oob_score.append(oob_score)

    dict_result = {"Train": list_results_train,
                   "Test": list_results_test,
                   "oob_score": list_oob_score,
                   "n_estimators": range_n_estimators}

    if save_mse:
        string_scaled = "_scaled" if scale else ""
        modelsaver.save_model(dict_result, f"rf_{min(range_n_estimators)}-{max(range_n_estimators)}"
                                           f"-results_train_test-{stockmodel}-{dict_option_types[option_type]}"
                                           f"-{max_features}{string_scaled}")

    return dict_result


def full_rf_all_model_columns_n_estimators():
    """
    Checks for max_features (["auto", "log2", 5]) and n_estimators of range(50, 751, 50).
    :return: None
    """
    max_features = ["auto", "log2", 5]
    for model in models:
        for column_fitting in columns_fitting:
            for max_feature in max_features:
                print(f"Model: {model} - {column_fitting} - {max_feature}")
                rf_n_estimators(stockmodel=model,
                                option_type=column_fitting,
                                range_n_estimators=range(50, 751, 50),
                                save_mse=True,
                                max_features=max_feature)
                if model == "BS" and column_fitting == "opt_standard":
                    print(f"Model: BS - Exacte - {max_feature}")
                    rf_n_estimators(stockmodel="BS",
                                    option_type="opt_exact_standard",
                                    range_n_estimators=range(50, 751, 50),
                                    save_mse=True,
                                    max_features=max_feature)


def rf_n_estimators_like_gpr(stockmodel="BS",
                             option_type="opt_exact_standard",
                             range_n_estimators=range(50, 751, 50),
                             save_mse=True,
                             max_features="auto",
                             scale=True):
    """
        Method to calculate the mse for a range of estimators

        :param stockmodel: str, "BS", "VG" or "H"
        :param option_type: str, "opt_standard", "opt_asianmean", "opt_lookbackmin" or "opt_lookbackmax"
                If stockmodel = "BS" -> "opt_exact_standard" is also possible
        :param range_n_estimators: list with the number of estimators for each run
        :param save_mse: bool, whenever to save all the values in a file.
        :param max_features: "auto", "log2" or a integer, for the splits in the Tree stockmodel
        :param scale: bool, if the data needs to be scaled or not
        :return: dict,with keys "Train", "Test", "oob_score", "n_estimators".
                Train = mse of the Training data
                Test = mse of the Test data
                oob_score = mse of the out-of-bag observations
                n_estimators = list of the number of estimators
        """

    dict_option_types = {"opt_exact_standard": "SE",
                         "opt_standard": "S",
                         "opt_asianmean": "A",
                         "opt_lookbackmin": "Lmin",
                         "opt_lookbackmax": "Lmax"}

    list_results_train = []
    list_results_test = []
    list_oob_score = []
    list_results_not_selected = []

    datamanager = dc.DataManager(stockmodel=stockmodel,
                                 option_type=option_type)

    n_samples = 10000
    random_state = 9943

    X_train, y_train, x_not_selected, y_not_selected = datamanager.get_random_training_data(n_samples=n_samples,
                                                                                            random_state=random_state,
                                                                                            get_not_selected_data=True)

    X_test, y_test = datamanager.get_test_data()

    if scale:
        scaler = preprocessing.StandardScaler().fit(X_train, y_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        x_not_selected = scaler.transform(x_not_selected)

    for n_estimator in range_n_estimators:
        rf_model = RandomForestRegressor(n_estimators=n_estimator,
                                         verbose=1,
                                         n_jobs=7,
                                         random_state=2458 + n_estimator,
                                         max_features=max_features,
                                         oob_score=True)
        rf_model.fit(X_train, y_train)

        mse_train = mean_squared_error(y_train, rf_model.predict(X_train))
        mse_test = mean_squared_error(y_test, rf_model.predict(X_test))
        oob_score = rf_model.oob_score_
        mse_test_not_selected = mean_squared_error(y_not_selected, rf_model.predict(x_not_selected))

        print(f'Train {mse_train}')
        print(f'Test {mse_test}')
        print(f'OOB score: {oob_score}')
        print(f'Not selected error: {mse_test_not_selected}')

        list_results_train.append(mse_train)
        list_results_test.append(mse_test)
        list_oob_score.append(oob_score)
        list_results_not_selected.append(mse_test_not_selected)

    dict_result = {"Train": list_results_train,
                   "Test": list_results_test,
                   "oob_score": list_oob_score,
                   "not_selected": list_results_not_selected,
                   "n_estimators": range_n_estimators}

    if save_mse:
        string_scaled = "_scaled" if scale else ""
        modelsaver.save_model(dict_result, f"rf_{min(range_n_estimators)}-{max(range_n_estimators)}"
                                           f"-results_train_test-{stockmodel}-{dict_option_types[option_type]}"
                                           f"-{max_features}{string_scaled}_likeGPR")

    return dict_result


def full_rf_all_model_columns_n_estimators_gpr():
    """
    Checks for max_features (["auto", "log2", 5]) and n_estimators of range(50, 751, 50).
    :return: None
    """
    max_features = ["auto", "log2", 5]
    for model in models:
        for column_fitting in columns_fitting:
            for max_feature in max_features:
                print(f"Model: {model} - {column_fitting} - {max_feature}")
                rf_n_estimators_like_gpr(stockmodel=model,
                                         option_type=column_fitting,
                                         range_n_estimators=range(50, 751, 50),
                                         save_mse=True,
                                         max_features=max_feature)
                if model == "BS" and column_fitting == "opt_standard":
                    print(f"Model: BS - Exacte - {max_feature}")
                    rf_n_estimators_like_gpr(stockmodel="BS",
                                             option_type="opt_exact_standard",
                                             range_n_estimators=range(50, 751, 50),
                                             save_mse=True,
                                             max_features=max_feature)

def one_tree_visualisation():
    rf = RandomForestRegressor(n_estimators=100, max_features="auto", n_jobs=6, verbose=2)

    datamanger = dc.DataManager()

    X, y = datamanger.get_training_data()

    # Train
    rf.fit(X, y)
    # Extract single tree
    estimator = rf.estimators_[8]

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    tree.plot_tree(rf.estimators_[8],
                   feature_names=X.columns,
                   max_depth=2,
                   filled=True)
    # plt.title("Random Forest: Decision Tree")
    fig.savefig('rf_individualtree.png')

    print(estimator.get_depth())


def full_dataset(stockmodel, option_type, only_call=False, with_percentage=False, scale=False):
    """
    print the results of the performance over the full dataset for the given stock stockmodel and option type
    :param stockmodel: str, "BS", "VG" or "H"
    :param option_type: str, "opt_standard", "opt_asianmean", "opt_lookbackmin" or
    :param only_call: bool (default=False), if the dataset only contains the call options
    :param with_percentage: bool (default=False),
            if the dataset needs to contain the percentage of the stock price and the strike
    :param scale: bool (default=False), if the dataset needs to be scaled
    """
    n_estimators = 700
    if (stockmodel == "BS" and option_type == "opt_standard") or stockmodel == "VG":
        max_feature = "log2"
    else:
        max_feature = 5

    dm = dc.DataManager(stockmodel=stockmodel,
                        option_type=option_type,
                        only_call=only_call,
                        with_percent=with_percentage)
    X_train, y_train = dm.get_training_data()

    if scale:
        scaler = preprocessing.StandardScaler().fit(X_train, y_train)
        X_train = scaler.transform(X_train)

    rf_model = RandomForestRegressor(n_jobs=8,
                                     verbose=0,
                                     max_features=max_feature,
                                     n_estimators=n_estimators)

    rf_model.fit(X_train, y_train)

    X_test, y_test = dm.get_test_data()
    if scale:
        X_test = scaler.transform(X_test)

    y_pred = rf_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred=y_pred)

    print(f"MSE: {mse}")


def main_full_dataset(only_call=False, with_percentage=False, scale=False):
    models = ["BS", "VG", "H"]
    option_types = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

    if only_call:
        print("Only for call options")

    for stockmodel in models:
        for option_type in option_types:
            print(f"{stockmodel}:{option_type}")
            full_dataset(stockmodel, option_type, only_call=only_call, with_percentage=with_percentage, scale=scale)


def train_real_data(symbol="SPX"):
    dm = dc.DataManagerRealData(symbol=symbol, test_month=9)
    X_train, y_train = dm.get_training_data()
    X_test, y_test = dm.get_test_data()

    rf_model = RandomForestRegressor(n_estimators=300,
                                     max_features="auto",
                                     n_jobs=8)

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"{symbol} - MSE= {mse}")


def main_real_data():
    for symbol in ["SPX", "SPXPM", "SX5E"]:
        print(symbol)
        train_real_data(symbol)


if __name__ == "__main__":
    print("Start RF")
    # full_rf_all_model_columns_n_estimators()

    # main_full_dataset(only_call=True, with_percentage=False, scale=True)
    # main_full_dataset(only_call=True, with_percentage=False, scale=False)
    # main_full_dataset(only_call=False, with_percentage=True, scale=True)
    # main_full_dataset(only_call=False, with_percentage=True, scale=False)
    # main_part_dataset_like_gpr(only_call=True)

    main_real_data()

    # full_rf_all_model_columns_n_estimators_gpr()

    # dm = dc.DataManager()
    # X_train, y_train = dm.get_training_data()
    # print(np.array(X_train).var())
