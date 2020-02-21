import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# todo bekijken om een verschil te maken met het standaard BS model, deze heeft andere namen voor de kolommen
# Which options that needs to be evaluated BS, VG, H
evaluate_stock_model = [False, True, True]
model_names = ["BS", "VG", "H"]
dict_model_names = {"BS": "Black Scholes",
                    "VG": "Variance Gamma",
                    "H": "Heston Model"}

# Which options that needs to be evaluated 'Standard', 'Asian','Lookback'
evaluate_options = [True, True, True]
option_names = ["Standard", "Asian", "Lookback"]

plot_mean = True
plot_min_max = False

plot_percentile = True
percentile = 2

restriction = None

# column name from the csv file as the X-variable
x_name = "paths"
# x_name = "time_step"

# column name from de csv file as Y-variable
y_name = "option_price"
y_label = "Price option"

dict_label_name = {"paths": "Number of paths",
                   "time_step": "Amount of steps per maturity"
                   }
x_label = dict_label_name[x_name]

# Standard title of the plot, first the option type and then the Stock model
title_plot = "Variance price {} option - {}"


########################################################################################################################
def read_data(filename):
    data = pd.read_csv(filename, header=0, comment='#')
    return data


def get_filename(model, option_type):
    return 'Datafiles/Test-steps and accuracy-{}-{}.csv'.format(model, option_type)


# todo toevoegen van een legende
def plot_change_variance(data, x_name, y_name, title, xlabel, ylabel, plot_min_max=False, plot_mean=False,
                         restriction=None, plot_percentile=False, percentile=2):
    data_x = data[x_name]
    data_y = data[y_name]

    if restriction is not None:
        data_y = data_y[restriction]
        data_x = data_x[restriction]

    unique_x = data_x.unique()

    if plot_min_max:
        min_line = []
        max_line = []
        for x in unique_x:
            min_line.append(np.min(data_y[data[x_name] == x]))
            max_line.append(np.max(data_y[data[x_name] == x]))
        plt.plot(unique_x, min_line, color='green')
        plt.plot(unique_x, max_line, color='green')

    if plot_percentile:
        percentile_max = 100 - percentile
        line_min = []
        line_max = []
        for x in unique_x:
            line_min.append(np.percentile(data_y[data[x_name] == x], percentile))
            line_max.append(np.percentile(data_y[data[x_name] == x], percentile_max))
        plt.plot(unique_x, line_min, color='orange')
        plt.plot(unique_x, line_max, color='orange')

    if plot_mean:
        mean_line = []
        for x in unique_x:
            mean_line.append(np.mean(data_y[data[x_name] == x]))
        plt.plot(unique_x, mean_line, color='red')

    plt.scatter(data_x, data_y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


########################################################################################################################


if any(evaluate_options) and any(evaluate_options):
    for evaluate_m, model in zip(evaluate_stock_model, model_names):
        for evaluate_opt, option in zip(evaluate_options, option_names):
            if evaluate_m and evaluate_opt:
                file_name = get_filename(model, option)
                data_options = read_data(file_name)

                plot_change_variance(data_options,
                                     x_name,
                                     y_name,
                                     title=title_plot.format(option, dict_model_names[model]),
                                     xlabel=x_label,
                                     ylabel=y_label,
                                     plot_mean=plot_mean,
                                     plot_min_max=plot_min_max,
                                     plot_percentile=plot_percentile,
                                     percentile=percentile,
                                     restriction=restriction)

# for data, title_name in zip(data_options, data_names):
#     plot_change_variance(data, x_name, y_name, "Variance price {} option - BS".format(title_name), dict_label_name[x_name],
#                          "Option price", plot_mean=True, plot_min_max=True)
#
# for data, title_name in zip(data_options, data_names):
#     print("Plot onder 5000")
#     plot_change_variance(data, x_name, y_name, "Variance price {} option - BS".format(title_name),
#                          dict_label_name[x_name],
#                          "Option price", plot_mean=True, plot_min_max=True, restriction=data["paths"] <= 5000)
#
#     print("Plot tussen 5000 en 10000")
#     plot_change_variance(data, x_name, y_name, "Variance price {} option - BS".format(title_name),
#                          dict_label_name[x_name],
#                          "Option price", plot_mean=True, plot_min_max=True,
#                          restriction=(data["paths"] > 5000) & (data["paths"] <= 10000))
#
#     print("Plot tussen 10000 en 15000")
#     plot_change_variance(data, x_name, y_name, "Variance price {} option - BS".format(title_name),
#                          dict_label_name[x_name],
#                          "Option price", plot_mean=True, plot_min_max=True,
#                          restriction=(data["paths"] > 10000) & (data["paths"] <= 15000))
#
#     print("Plot tussen 15000 en 20000")
#     plot_change_variance(data, x_name, y_name, "Variance price {} option - BS".format(title_name),
#                          dict_label_name[x_name],
#                          "Option price", plot_mean=True, plot_min_max=True,
#                          restriction=(data["paths"] > 15000) & (data["paths"] <= 20000))
#
# # print(data_asian["paths"] <= 5000)
# # test_data = data_asian[y_name]
# # test = test_data[data_asian["paths"] <= 5000]
# # # print(test)
#
# ########################################################################################################################
# # ------------------------------- Black Scholes -----------------------------------------------------------------------#
# ########################################################################################################################
# if evaluate_BS:
#     filename = 'Datafiles/Test-steps and accuracy-BS-v1.csv'
#     show_plots = True
#
#     data_bs = read_data(filename)
#
#     number_paths = data_bs['paths'].unique()
#     step_sizes = data_bs['time_step'].unique()
#
#
#     def plot_accuracy_graph(values, x_name, y_name, title, name_x_axis, name_y_axis,
#                             fix_locations_name=None, fixed_value_locations=None):
#         locations = [i for i in range(len(values[x_name]))]
#         if fix_locations_name is not None:
#             if fixed_value_locations is not None:
#                 locations = data_bs[fix_locations_name] == fixed_value_locations
#             else:
#                 raise ValueError
#
#         x_values = values[x_name][locations]
#         y_values = values[y_name][locations]
#
#         unique_x = x_values.unique()
#         mean_y = []
#         for x in unique_x:
#             mean_y.append(np.mean(y_values[values[x_name] == x]))
#
#         plt.scatter(x_values, y_values)
#         plt.xlabel(name_x_axis)
#         plt.ylabel(name_y_axis)
#         plt.title(title)
#         plt.plot(unique_x, mean_y, color='red')
#         plt.show()
#
#
#     if show_plots:
#         # Test function:
#         plot_accuracy_graph(data_bs, "paths", "accuracy_normal", "Performance", "Number of paths",
#                             "Relative Difference")
#         plt.show()
#
#         plot_accuracy_graph(data_bs, "paths", "accuracy_absolute", "Performance", "Number of paths",
#                             "Absolute Relative Difference")
#         plt.show()
#
#         plot_accuracy_graph(data_bs, "time_step", "accuracy_absolute", "Performance", "Number of steps",
#                             "Absolute Relative Difference")
#         plt.show()
#
#         plot_accuracy_graph(data_bs, "time_step", "accuracy_normal", "Performance", "Number of steps",
#                             "Relative Difference")
#         plt.show()
#
#         plot_accuracy_graph(data_bs, "time_step", "time", "Time", "Number of steps",
#                             "Time")
#         plt.show()
#
#         plot_accuracy_graph(data_bs, "paths", "time", "Time", "Amount of paths",
#                             "Time")
#         plt.show()
