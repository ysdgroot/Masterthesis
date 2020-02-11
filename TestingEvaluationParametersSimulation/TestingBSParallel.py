import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from sklearn.linear_model import LinearRegression


def read_data(filename):
    data = pd.read_csv(filename, header=0, comment='#')
    return data


filename = 'Test-steps and accuracy-BS-v3.csv'
data = read_data(filename)

for i in range(1000, 20001, 1000):
    print(len(data[data["paths"] == i]))

number_paths = data['paths'].unique()
step_sizes = data['time_step'].unique()


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

# plot_accuracy_graph(data, "accuracy_normal", "paths", "Performance", "Number of paths",
#                         "Relative Difference")
#
# plot_accuracy_graph(data, "accuracy_absolute", "paths", "Performance", "number of paths",
#                         "Absolute Relative Difference")
#
# plot_accuracy_graph(data, "accuracy_normal", "time_step", "Performance", "Number of steps",
#                         "Maximum Absolute Relative Difference")
#
# plot_accuracy_graph(data, "accuracy_absolute", "time_step", "Performance", "Number of steps",
#                         "Absolute Relative Difference")

# Algemene 'conclusie' is dat het aantal paden belangrijker is dan de stapgrootte
# Van hieruit gaan we kiezen voor 20000paden en voor stapgrootte 200
