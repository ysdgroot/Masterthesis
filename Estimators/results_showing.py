import modelsaver
from Estimators import train_test_rf
import importlib.resources as pkg_resources
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

models = ["BS", "VG", "H"]
columns_fitting = ["opt_standard", "opt_asianmean", "opt_lookbackmin", "opt_lookbackmax"]

dict_column_to_option = {"opt_standard": "Standard",
                         "opt_asianmean": "Asian",
                         "opt_lookbackmin": "Lookback (min)",
                         "opt_lookbackmax": "Lookback (max)",
                         "opt_exact_standard": "Standard(theory)"}

code_options = ["S", "A", "Lmin", "Lmax"]


# model = models[2]
# column_fitting = columns_fitting[1]


def plot_results(ml_model, stockmodel, column_fitting, dict_results, list_estimators, save_plot=False):
    # todo; comments
    list_names_fig = []

    figure, ax = plt.subplots()

    for key, value in dict_results.items():
        fig, = ax.plot(list_estimators, value)
        # list_figures.append(fig)
        list_names_fig.append(key)

    ax.set_title(f"Performance {ml_model}-{stockmodel}-{dict_column_to_option[column_fitting]} option")
    ax.set_ylabel("Mean squared error")
    ax.set_xlabel("Number of estimators")
    ax.legend(list_names_fig)

    if save_plot:
        figure.savefig(f"{ml_model}-{stockmodel}-{dict_column_to_option[column_fitting]}.png")

    plt.show()


def rf_plot_train_test(model, column_fitting, save_plot=False):
    """

    :param model:
    :param column_fitting:
    :param save_plot:
    :return:
    """
    # todo: comments
    max_features = ["auto", "log2", 5]
    dict_codes = {"opt_standard": "S",
                  "opt_asianmean": "A",
                  "opt_lookbackmin": "Lmin",
                  "opt_lookbackmax": "Lmax",
                  "opt_exact_standard": "SE"}

    opt_type_code = dict_codes[column_fitting]

    base_file_name = "rf_50-1000-results_train_test-{0}-{1}-{2}.p"

    pickle_files = [base_file_name.format(model, opt_type_code, feature) for feature in max_features]
    file_names = [pkg_resources.open_text(train_test_rf, pickle_file).name for pickle_file in pickle_files]

    dict_values = [modelsaver.get_model(file_name) for file_name in file_names]

    dict_plotting = {}
    for feature, results in zip(max_features, dict_values):
        dict_plotting[feature] = results['Test']

    plot_results("RF", model, column_fitting, dict_plotting, range(50, 1001, 50), save_plot=save_plot)


def rf_plot_all_train_test(save_plots=False):
    for model in models:
        for column_fitting in columns_fitting:
            rf_plot_train_test(model, column_fitting, save_plot=save_plots)


def plotting_results_cv_svr():
    # todo: verder aan werken
    dict_cv_results = modelsaver.get_model("SVR-random_search.p").cv_results_

    ranks = dict_cv_results['rank_test_neg_mean_squared_error']
    best_positions = np.where(ranks <= 1)

    # print(best_positions)

    print(np.array(dict_cv_results['params'])[best_positions])
    print(dict_cv_results['mean_test_neg_mean_squared_error'][best_positions])
    # beste resultaten zijn een poly met graad 2 (mse=125) nadien rbf (mse=254)

    print(dict_cv_results)
    return None


def plotting_results_cv_gpr():
    return None


def results_nn_optimizers_activations():
    # data_opt = pd.read_csv("optimizers-activation-nodes-new.csv", comment='#', header=0)
    data_opt = pd.read_csv("optimizers-activation-nodes-v3-newversion_input6.csv", comment='#', header=0)

    print("The top 5 for each node")
    top5_per_node = data_opt.sort_values(["n_nodes", "mse"], ascending=True).groupby('n_nodes').head(5)
    print(top5_per_node)

    print("Top 5 for each optimizer")
    best_optimizer = data_opt.sort_values(["optimizer", "mse"], ascending=True).groupby('optimizer').mean()
    print(best_optimizer)

    print("Top 5 for each activationfunction")
    best_optimizer = data_opt.sort_values(["activation", "mse"], ascending=True).groupby('activation').mean()
    print(best_optimizer)

    # print("All adamax values")
    # print(data_opt.loc[(data_opt['mse'] < 100)].groupby('optimizer').mean())

    return None


if __name__ == '__main__':
    print("Start")
    # plotting_results_cv_rf("H", "opt_standard", measure="mse")
    # todo zoek de ranking voor de random forests en bepaal dan de precieze mse van het totale
    # results_nn_optimizers_activations()
    # plotting_results_cv_svr()
    rf_plot_all_train_test(False)
