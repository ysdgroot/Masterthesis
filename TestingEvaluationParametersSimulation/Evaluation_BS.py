import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_data(filename):
    data = pd.read_csv(filename, header=0, comment='#')
    return data


filename = 'Test-steps and accuracy-BS-v1.csv'
data = read_data(filename)

number_paths = data['paths'].unique()
step_sizes = data['time_step'].unique()
show_plots = True


def plot_accuracy_graph(values, x_name, y_name, title, name_x_axis, name_y_axis,
                        fix_locations_name=None, fixed_value_locations=None):
    locations = [i for i in range(len(values[x_name]))]
    if fix_locations_name is not None:
        if fixed_value_locations is not None:
            locations = data[fix_locations_name] == fixed_value_locations
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
    plot_accuracy_graph(data, "paths", "accuracy_normal", "Performance", "Number of paths",
                        "Relative Difference")
    plt.show()

    plot_accuracy_graph(data, "paths", "accuracy_absolute", "Performance", "Number of paths",
                        "Absolute Relative Difference")
    plt.show()

    plot_accuracy_graph(data, "time_step", "accuracy_absolute", "Performance", "Number of steps",
                        "Absolute Relative Difference")
    plt.show()

    plot_accuracy_graph(data, "time_step", "accuracy_normal", "Performance", "Number of steps",
                        "Relative Difference")
    plt.show()

    plot_accuracy_graph(data, "time_step", "time", "Time", "Number of steps",
                        "Time")
    plt.show()

    plot_accuracy_graph(data, "paths", "time", "Time", "Amount of paths",
                        "Time")
    plt.show()
