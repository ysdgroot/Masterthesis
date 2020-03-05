import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Which option_types that needs to be evaluated BS, VG, H
evaluate_stock_model = [False, False, False]
model_names = ["BS", "VG", "H"]
dict_model_names = {"BS": "Black Scholes",
                    "VG": "Variance Gamma",
                    "H": "Heston"}

# Which option_types that needs to be evaluated 'Standard', 'Asian','Lookback'
evaluate_options = [True, True, True]
option_names = ["Standard", "Asian", "Lookback"]

plot_mean = True
plot_min_max = False

plot_percentile = False
percentile = 1

# restriction = ("paths", 15000, 20000)
# restriction = ("time_step", 0, 500)
restriction = None

# column name from the csv file as the X-variable
x_name = "paths"
# x_name = "time_step"

# column name from de csv file as Y-variable
y_name = "option_price"
y_label = "Price option"

# accuracy_absolute,accuracy_normal,exact_value
y_name_standard_BS = "accuracy_normal"
# y_name_standard_BS = "accuracy_absolute"
dict_y_label_standardBS = {"accuracy_normal": "Relative difference",
                           "accuracy_absolute": "Absolute Relative difference"}

dict_label_name = {"paths": "Number of paths",
                   "time_step": "Amount of steps per maturity"
                   }
x_label = dict_label_name[x_name]

# Standard title of the plot, first the option type and then the Stock model
title_plot = "Variance price {} option - {}"
title_plot_standard_BS = "Performance Simulation vs Theory - {}"

dict_title_restriction = {"paths": "-Paths=({},{})",
                          "time_step": "-Steps=({},{})"
                          }


########################################################################################################################
def read_data(filename):
    data = pd.read_csv(filename, header=0, comment='#')
    return data


def get_filename(model, option_type):
    return f'Datafiles/{dict_model_names[model]}/Test-steps and accuracy-{model}-{option_type}.csv'


# todo toevoegen van een legende
def plot_change_variance(data, x_name, y_name, title, xlabel, ylabel, plot_min_max=False, plot_mean=False,
                         restriction=None, plot_percentile=False, percentile=1):
    data_x = data[x_name]
    data_y = data[y_name]

    unique_x = data_x.unique()

    if restriction is not None:
        data_y = data_y[(restriction[1] <= data[restriction[0]]) & (data[restriction[0]] <= restriction[2])]
        data_x = data_x[(restriction[1] <= data[restriction[0]]) & (data[restriction[0]] <= restriction[2])]

        # resiction of the data for later use; otherwise there will be indexing problems.
        data = data[(restriction[1] <= data[restriction[0]]) & (data[restriction[0]] <= restriction[2])]

        title += dict_title_restriction[restriction[0]].format(restriction[1], restriction[2])

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
    # todo deze stuk code verwijderen
    # restrictions = [("paths", 1000, 5000), ("paths", 5000, 10000), ("paths", 10000, 15000), ("paths", 15000, 20000)]
    # restrictions = [("paths", i, i + 1000) for i in range(1000, 20000, 1000)]
    restrictions = [("time_step", 5, 50), ("time_step", 60, 100), ("time_step", 200, 400), ("time_step", 900, 1000)]
    # # ("time_step", 101, 300), ("time_step", 301, 600),, ("time_step", 801, 1000), ("time_step", 55, 100), ("time_step", 101, 300),
    for restrict in restrictions:
        plt.scatter(data_x[(restrict[1] <= data[restrict[0]]) & (data[restrict[0]] <= restrict[2])],
                    data_y[(restrict[1] <= data[restrict[0]]) & (data[restrict[0]] <= restrict[2])])

    # plt.scatter(data_x, data_y)
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

                # special case BS and Standard option
                if model == "BS" and option == "Standard":
                    plot_change_variance(data_options,
                                         x_name,
                                         y_name_standard_BS,
                                         title=title_plot_standard_BS.format(dict_model_names[model]),
                                         xlabel=x_label,
                                         ylabel=dict_y_label_standardBS[y_name_standard_BS],
                                         plot_mean=plot_mean,
                                         plot_min_max=plot_min_max,
                                         plot_percentile=plot_percentile,
                                         percentile=percentile,
                                         restriction=restriction)
                else:
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

########################################################################################################################
# TESTING ########################
##################################
model = "VG"
option = "Standard"

file_name = "Datafiles/Test-steps and accuracy-VG-v1-1.csv"
# file_name = "Datafiles/Test-steps and accuracy-H-v2-1-Asian.csv"
# file_name = "Datafiles/Test-steps and accuracy-H-v3-1-Lookback.csv"
file_name_2 = get_filename(model, option)

data_options = read_data(file_name)
data_options2 = read_data(file_name_2)

# new_data = data_options.append(data_options2)
new_data = data_options

# print(new_data)

x_name = "paths"
# x_name = "time_step"
# restriction = ("paths", 10000, 10000)
#
# restrictions = [("paths", 1000, 5000), ("paths", 5000, 10000), ("paths", 10000, 15000), ("paths", 15000, 20000)]

# y_name = "variance"

# restriction = ("time_step", 5, 100)

plot_change_variance(new_data, x_name,
                     y_name,
                     title=title_plot.format(option, dict_model_names[model]),
                     xlabel=x_label,
                     ylabel=y_label,
                     plot_mean=plot_mean,
                     plot_min_max=plot_min_max,
                     plot_percentile=False,
                     percentile=percentile,
                     restriction=restriction)

# unique_time_steps = data_options["time_step"].unique()
# unique_paths = data_options["paths"].unique()
#
# n_paths = 15000
# for path in unique_paths:
#     var_all_time_steps = []
#     if 1:
#         time_steps = unique_time_steps[unique_time_steps <= 100]
#         for i in time_steps:
#             positions_paths = data_options["paths"] == path
#             positions_time_steps = data_options["time_step"] == i
#             var_all_time_steps.append(np.var(data_options[positions_paths & positions_time_steps]["option_price"]))
#
#         plt.plot(time_steps, var_all_time_steps)
#
# plt.show()

# restrictions = [("paths", 1000, 5000)] + [("paths", i, i + 5000) for i in range(5000, 20000, 5000)]
#
# for restrict in restrictions:
#     plot_change_variance(new_data, x_name,
#                          y_name,
#                          title=title_plot.format(option, dict_model_names[model]),
#                          xlabel=x_label,
#                          ylabel=y_label,
#                          plot_mean=plot_mean,
#                          plot_min_max=plot_min_max,
#                          plot_percentile=True,
#                          percentile=0,
#                          restriction=restrict)
