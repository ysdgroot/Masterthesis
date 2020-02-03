import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from sklearn.linear_model import LinearRegression


def read_data(filename):
    data = pd.read_csv(filename, header=0, comment='#')
    return data


filename = 'Test-steps and accuracy-BS-v1.csv'
data = read_data(filename)

number_paths = data['paths'].unique()
step_sizes = data['time_step'].unique()
lst_legend = []
show_plots = False


def plot_accuracy_graph(values, accuracy_type, x_name, title, name_x_axis, name_y_axis,
                        fix_locations_name=None, fixed_value_locations=None):
    # todo: maken van grafiek met een fixed value (bv path == 100)
    locations = [i for i in range(len(values[x_name]))]
    if fix_locations_name is not None:
        if fixed_value_locations is not None:
            locations = data[fix_locations_name] == fixed_value_locations
        else:
            raise ValueError

    x_values = values[x_name][locations]
    y_values = values[accuracy_type][locations]

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
    plot_accuracy_graph(data, "accuracy_normal", "paths", "Performance", "Number of paths",
                        "Relative Difference")

    plot_accuracy_graph(data, "accuracy_absolute", "paths", "Performance", "number of paths",
                        "Absolute Relative Difference")
    # plt.show()

    plot_accuracy_graph(data, "max_abs_diff", "paths", "Performance", "number of paths",
                        "Maximum Absolute Relative Difference")
    # plt.show()

    plot_accuracy_graph(data, "max_abs_diff", "time_step", "Performance", "Number of steps",
                        "Maximum Absolute Relative Difference")
    # plt.show()

    plot_accuracy_graph(data, "accuracy_absolute", "time_step", "Performance", "Number of steps",
                        "Absolute Relative Difference")
    # plt.show()

    # plot_accuracy_graph(data, "min_rel_diff", "time_step")
    # plot_accuracy_graph(data, "max_rel_diff", "time_step")
    # unique_paths = data["paths"].unique()
    # for path in unique_paths:
    #     plot_accuracy_graph(data, "accuracy_normal", "time_step", "Performance", "Number of steps",
    #                         "Relative difference",
    #                         fix_locations_name='paths',
    #                         fixed_value_locations=path)
    # plt.show()

x_values = "paths"
# x_values = "time_step"

# y_values = "accuracy_normal"
# y_values = "accuracy_absolute"
y_values = "time"

print(data["time"])

# # print(data[x_values])
# # print(data[y_values])
plt.scatter(data[x_values], data[y_values])
plt.show()

# # make graph for each path size vs accuracy
# for path in number_paths:
#     time_steps = data['time_step'][data['paths'] == path]
#     # accuracy_abs = data['accuracy_absolute'][data['paths'] == path]
#     # p, = plt.plot(time_steps, accuracy_abs)
#     accuracy = data['accuracy_normal'][data['paths'] == path]
#     p, = plt.plot(time_steps, accuracy)
#     lst_legend.append(p)
# plt.legend(lst_legend, number_paths)
# plt.show()
#
# # make grapf for each time step vs accuracy
# for stepsize in step_sizes:
#     locations = data['time_step'] == stepsize
#     paths = data['paths'][locations]
#     accuracy = data['accuracy_normal'][locations]
#     p, = plt.plot(paths, accuracy)
#
# plt.show()

# plt.plot(data["time_step"], data["accuracy_normal"])
# plt.show()
