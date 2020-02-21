import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_data(filename):
    data = pd.read_csv(filename, header=0, comment='#')
    return data


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


print("Start of the Standard/Asian/Lookback functionality Heston Model")

file_name_standard = 'Datafiles/Test-steps and accuracy-H-v1.csv'
file_name_asian = 'Datafiles/Test-steps and accuracy-H-v2-Asian.csv'
file_name_lookback = 'Datafiles/Test-steps and accuracy-H-v3-Lookback.csv'

data_standard = read_data(file_name_standard)
data_asian = read_data(file_name_asian)
data_lookback = read_data(file_name_lookback)

data_options = [data_standard, data_asian, data_lookback]

# column name from the csv file as the X-variable
x_name = "paths"
# x_name = "time_step"

# column name from de csv file as Y-variable
y_name = "option_price"

dict_label_name = {"paths": "Number of paths",
                   "time_step": "Amount of steps per maturity"
                   }

data_names = ["Standard", "Asian", "Lookback"]

for data, title_name in zip(data_options, data_names):
    plot_change_variance(data, x_name, y_name, "Variance price {} option - H".format(title_name),
                         dict_label_name[x_name],
                         "Option price", plot_mean=True, plot_min_max=True, plot_percentile=True)

# todo: Dit gebruiken om op een structurele manier ieder geval te bekijken
# for data, title_name in zip(data_options, data_names):
#     print("Plot onder 5000")
#     plot_change_variance(data, x_name, y_name, "Variance price {} option - VG".format(title_name),
#                          dict_label_name[x_name],
#                          "Option price", plot_mean=True, plot_min_max=True, restriction=data["paths"] <= 5000)
#
#     print("Plot tussen 5000 en 10000")
#     plot_change_variance(data, x_name, y_name, "Variance price {} option - VG".format(title_name),
#                          dict_label_name[x_name],
#                          "Option price", plot_mean=True, plot_min_max=True, restriction=(data["paths"] > 5000) & (data["paths"] <= 10000))
#
#     print("Plot tussen 10000 en 15000")
#     plot_change_variance(data, x_name, y_name, "Variance price {} option - VG".format(title_name),
#                          dict_label_name[x_name],
#                          "Option price", plot_mean=True, plot_min_max=True,
#                          restriction=(data["paths"] > 10000) & (data["paths"] <= 15000))
#
#     print("Plot tussen 15000 en 20000")
#     plot_change_variance(data, x_name, y_name, "Variance price {} option - VG".format(title_name),
#                          dict_label_name[x_name],
#                          "Option price", plot_mean=True, plot_min_max=True,
#                          restriction=(data["paths"] > 15000) & (data["paths"] <= 20000))

# number_paths = data['paths'].unique()
# step_sizes = data['time_step'].unique()
# show_plots = False
#
# print(number_paths)
#
# variance_global_paths = []
# for path in number_paths:
#     variance_global_paths.append(np.var(data["option_value"][data["paths"] == path]))
#
# print(variance_global_paths)
#
# print(step_sizes)
# variance_global_step = []
# for step_size in step_sizes:
#     variance_global_step.append(np.var(data["option_value"][data["time_step"] == step_size]))
#
# print(variance_global_step)
#
# variance_matrix = []
# for path in number_paths:
#     variance_paths = list()
#     for step_size in step_sizes:
#         steps = data["time_step"] == step_size
#         pad = data["paths"] == path
#         variance_paths.append(np.var(data["option_value"][steps & pad]))
#     variance_matrix.append(variance_paths)
#
# for row in variance_matrix:
#     print(row)
