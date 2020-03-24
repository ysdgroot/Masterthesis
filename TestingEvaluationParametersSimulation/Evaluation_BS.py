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
plot_min_max = True

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


def plot_change_variance(data, x_name, y_name, title, xlabel, ylabel, plot_min_max=False, plot_mean=False,
                         restriction=None, plot_percentile=False, percentile=1):
    data_x = data[x_name]
    data_y = data[y_name]

    legend_variables = []
    legend_variable_names = []

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
        line_min, = plt.plot(unique_x, min_line, color='green')
        plt.plot(unique_x, max_line, color='green')

        # the lines are the same for the minimum and maximum
        legend_variables.append(line_min)
        legend_variable_names.append("Minimum and maximum")

    if plot_percentile:
        percentile_max = 100 - percentile
        line_min = []
        line_max = []
        for x in unique_x:
            line_min.append(np.percentile(data_y[data[x_name] == x], percentile))
            line_max.append(np.percentile(data_y[data[x_name] == x], percentile_max))
        line_min_percentile, = plt.plot(unique_x, line_min, color='orange')
        plt.plot(unique_x, line_max, color='orange')

        # the lines are the same for the minimum and maximum percentile
        legend_variables.append(line_min_percentile)
        legend_variable_names.append(f"Percentiles {percentile} - {percentile_max}")

    if plot_mean:
        mean_line = []
        for x in unique_x:
            mean_line.append(np.mean(data_y[data[x_name] == x]))
        line_mean, = plt.plot(unique_x, mean_line, color='red')

        legend_variables.append(line_mean)
        legend_variable_names.append("Mean")
    # todo deze stuk code verwijderen
    # restrictions = [("paths", 1000, 5000), ("paths", 5000, 10000), ("paths", 10000, 15000), ("paths", 15000, 20000)]
    # restrictions = [("paths", i, i + 1000) for i in range(1000, 20000, 1000)]
    restrictions = [("time_step", 5, 50), ("time_step", 60, 100), ("time_step", 200, 400), ("time_step", 900, 1000)]
    # # ("time_step", 101, 300), ("time_step", 301, 600),, ("time_step", 801, 1000), ("time_step", 55, 100), ("time_step", 101, 300),
    for restrict in restrictions:
        scat_points = plt.scatter(data_x[(restrict[1] <= data[restrict[0]]) & (data[restrict[0]] <= restrict[2])], data_y[(restrict[1] <= data[restrict[0]]) & (data[restrict[0]] <= restrict[2])])

        legend_variables.append(scat_points)
        legend_variable_names.append(f"{restrict[0]}: [{restrict[1]}-{restrict[2]}]")

    # plt.scatter(data_x, data_y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend(legend_variables, legend_variable_names)

    plt.show()


def test_bar_plot(data,
                  title="",
                  xlabel="",
                  ylabel="",
                  n_splits=4):
    x_name = "time_step"
    y_name = "option_price"

    x_name = "paths"

    data_x = data[x_name]
    data_y = data[y_name]

    labels = data_x.unique() // 1000

    list_mean_1 = []
    list_mean_2 = []
    list_mean_3 = []
    list_mean_4 = []

    for time_step in data_x.unique():
        positions_time_step = data_x == time_step

        # list_mean_1.append(np.var(data_y[positions_time_step & (data["paths"] >= 1000) & (data["paths"] <= 5000)]))
        # list_mean_2.append(np.var(data_y[positions_time_step & (data["paths"] >= 6000) & (data["paths"] <= 10000)]))
        # list_mean_3.append(np.var(data_y[positions_time_step & (data["paths"] >= 11000) & (data["paths"] <= 15000)]))
        # list_mean_4.append(np.var(data_y[positions_time_step & (data["paths"] >= 16000) & (data["paths"] <= 20000)]))

        list_mean_1.append(np.var(data_y[positions_time_step & (data["time_step"] >= 5) & (data["time_step"] <= 20)]))
        list_mean_2.append(np.var(data_y[positions_time_step & (data["time_step"] >= 5) & (data["time_step"] <= 50)]))
        list_mean_3.append(np.var(data_y[positions_time_step & (data["time_step"] >= 55) & (data["time_step"] <= 75)]))
        list_mean_4.append(
            np.var(data_y[positions_time_step & (data["time_step"] >= 900) & (data["time_step"] <= 1000)]))

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3 * width / 2, list_mean_1, width, label='klasse 1')
    rects2 = ax.bar(x - width / 2, list_mean_2, width, label='klasse 2')
    rects3 = ax.bar(x + width / 2, list_mean_3, width, label='klasse 3')
    rects4 = ax.bar(x + 3 * width / 2, list_mean_4, width, label='klasse 4')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Option price')
    ax.set_xlabel('Number of paths')
    ax.set_title('Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # ax.set_ylim([31, 35])

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

test_bar_plot(new_data)

# print(new_data)

x_name = "paths"
# x_name = "time_step"
# restriction = ("paths", 10000, 10000)
#
# restrictions = [("paths", 1000, 5000), ("paths", 5000, 10000), ("paths", 10000, 15000), ("paths", 15000, 20000)]

# y_name = "variance"

# restriction = ("time_step", 5, 100)

# plot_change_variance(new_data, x_name,
#                      y_name,
#                      title=title_plot.format(option, dict_model_names[model]),
#                      xlabel=x_label,
#                      ylabel=y_label,
#                      plot_mean=plot_mean,
#                      plot_min_max=plot_min_max,
#                      plot_percentile=False,
#                      percentile=percentile,
#                      restriction=restriction)

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
