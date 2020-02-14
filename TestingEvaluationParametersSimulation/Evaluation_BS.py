import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

evaluate_BS = False
evaluate_VG = False
evaluate_H = False


def read_data(filename):
    data = pd.read_csv(filename, header=0, comment='#')
    return data


def plot_change_variance(data, x_name, y_name, title, xlabel, ylabel, plot_min_max=False, plot_mean=False):
    data_x = data[x_name]
    data_y = data[y_name]
    unique_x = data_x.unique()

    if plot_min_max:
        min_line = []
        max_line = []
        for x in unique_x:
            min_line.append(np.min(data[y_name][data[x_name] == x]))
            max_line.append(np.max(data[y_name][data[x_name] == x]))
        plt.plot(unique_x, min_line, color='green')
        plt.plot(unique_x, max_line, color='green')

    if plot_mean:
        mean_line = []
        for x in unique_x:
            mean_line.append(np.mean(data[y_name][data[x_name] == x]))
        plt.plot(unique_x, mean_line, color='red')

    plt.scatter(data_x, data_y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


file_n = 'Test-steps and accuracy-BS-v2-Asian.csv'
data = read_data(file_n)

x_name = "paths"
# x_name = "time_step"

dict_label_name = {"paths": "Number of paths",
                   "time_step": "Amount of steps per maturity"
                   }

plot_change_variance(data, x_name, "option_price", "Variance price Asian option - BS", dict_label_name[x_name],
                     "Option price", plot_mean=True, plot_min_max=True)

########################################################################################################################
# ------------------------------- Black Scholes -----------------------------------------------------------------------#
########################################################################################################################
if evaluate_BS:
    filename = 'Test-steps and accuracy-BS-v1.csv'
    show_plots = True

    data_bs = read_data(filename)

    number_paths = data_bs['paths'].unique()
    step_sizes = data_bs['time_step'].unique()


    def plot_accuracy_graph(values, x_name, y_name, title, name_x_axis, name_y_axis,
                            fix_locations_name=None, fixed_value_locations=None):
        locations = [i for i in range(len(values[x_name]))]
        if fix_locations_name is not None:
            if fixed_value_locations is not None:
                locations = data_bs[fix_locations_name] == fixed_value_locations
            else:
                raise ValueError

        x_values = values[x_name][locations]
        y_values = values[y_name][locations]

        unique_x = x_values.unique()
        mean_y = []
        for x in unique_x:
            mean_y.append(np.mean(y_values[values[x_name] == x]))

        plt.scatter(x_values, y_values)
        plt.xlabel(name_x_axis)
        plt.ylabel(name_y_axis)
        plt.title(title)
        plt.plot(unique_x, mean_y, color='red')
        plt.show()


    if show_plots:
        # Test function:
        plot_accuracy_graph(data_bs, "paths", "accuracy_normal", "Performance", "Number of paths",
                            "Relative Difference")
        plt.show()

        plot_accuracy_graph(data_bs, "paths", "accuracy_absolute", "Performance", "Number of paths",
                            "Absolute Relative Difference")
        plt.show()

        plot_accuracy_graph(data_bs, "time_step", "accuracy_absolute", "Performance", "Number of steps",
                            "Absolute Relative Difference")
        plt.show()

        plot_accuracy_graph(data_bs, "time_step", "accuracy_normal", "Performance", "Number of steps",
                            "Relative Difference")
        plt.show()

        plot_accuracy_graph(data_bs, "time_step", "time", "Time", "Number of steps",
                            "Time")
        plt.show()

        plot_accuracy_graph(data_bs, "paths", "time", "Time", "Amount of paths",
                            "Time")
        plt.show()

########################################################################################################################
# ------------------------------- Variance Gamma ----------------------------------------------------------------------#
########################################################################################################################
if evaluate_VG:
    filename = 'Test-steps and accuracy-VG-v1.csv'
    data_vg = read_data(filename)

    number_paths = data_vg['paths'].unique()
    step_sizes = data_vg['time_step'].unique()
    show_plots = False

    print(number_paths)

    variance_global_paths = []
    for path in number_paths:
        variance_global_paths.append(np.var(data_vg["option_value"][data_vg["paths"] == path]))

    print(variance_global_paths)

    print(step_sizes)
    variance_global_step = []
    for step_size in step_sizes:
        variance_global_step.append(np.var(data_vg["option_value"][data_vg["time_step"] == step_size]))

    print(variance_global_step)

    variance_matrix = []
    for path in number_paths:
        variance_paths = list()
        for step_size in step_sizes:
            steps = data_vg["time_step"] == step_size
            pad = data_vg["paths"] == path
            variance_paths.append(np.var(data_vg["option_value"][steps & pad]))
        variance_matrix.append(variance_paths)

    for row in variance_matrix:
        print(row)

########################################################################################################################
# ------------------------------- Heston ------------------------------------------------------------------------------#
########################################################################################################################
if evaluate_H:
    print("Werk aan het evalueren van het Heston model")
