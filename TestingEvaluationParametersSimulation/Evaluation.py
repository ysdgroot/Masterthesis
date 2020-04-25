import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Which option_types that needs to be evaluated BS, VG, H
evaluate_stock_model = [True, True, True]
model_names = ["BS", "VG", "H"]
dict_model_names = {"BS": "Black Scholes",
                    "VG": "Variance Gamma",
                    "H": "Heston"}

# Which option_types that needs to be evaluated 'Standard', 'Asian','Lookback'
option_names = ["Standard", "Asian", "Lookback"]
dict_option_names_nl = dict(zip(option_names, ["Standaard", "Aziatische", "Lookback"]))
evaluate_options = [True, True, True]

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
y_label = "Prijs optie"

y_name_standard_BS = "relative_diff"
# y_name_standard_BS = "absolute_diff"
dict_y_label_standardBS = {"relative_diff": "Relatief verschil",
                           "absolute_diff": "Absoluut relatief verschil"}

dict_label_name = {"paths": "Aantal paden",
                   "time_step": "Aantal stappen per maturiteit"
                   }
x_label = dict_label_name[x_name]

# Standard title of the plot, first the option type and then the Stock model
title_plot = "Variantie prijs {} optie - {}"
title_plot_standard_BS = "Prestatie Simulatie vs Theorie - {}"

dict_title_restriction = {"paths": "-Paden=({},{})",
                          "time_step": "-Stappen=({},{})"
                          }


########################################################################################################################
def read_data(model, option_type):
    """
    Read and return the data from a model and option type.

    :param model: str (BS, VG or H)
    :param option_type: str, ("Standard", "Asian", "Lookback")
    :return: pd.Dataframe (columns depends on the model)
    """
    filename = get_filename(model, option_type)
    data = pd.read_csv(filename, header=0, comment='#')
    return data


def get_filename(model, option_type):
    """
    Return the filename where the data is stored.
    :param model: str (BS, VG or H)
    :param option_type: str, ("Standard", "Asian", "Lookback(min)", "Lookback(max)")
    :return: str
    """
    return f'Datafiles/{dict_model_names[model]}/Test-steps and accuracy-{model}-{option_type}.csv'


def plot_change_variance(data, x_name, y_name, title, xlabel, ylabel, plot_min_max=False, plot_mean=False,
                         restriction=None, plot_percentile=False, percentile=1):
    """
    # todo: comments schrijven!
    :param data:
    :param x_name:
    :param y_name:
    :param title:
    :param xlabel:
    :param ylabel:
    :param plot_min_max:
    :param plot_mean:
    :param restriction:
    :param plot_percentile:
    :param percentile:
    :return:
    """
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
        line_min, = plt.plot(unique_x, min_line, color='orange')
        plt.plot(unique_x, max_line, color='orange')

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
        line_min_percentile, = plt.plot(unique_x, line_min, color='green')
        plt.plot(unique_x, line_max, color='green')

        # the lines are the same for the minimum and maximum percentile
        legend_variables.append(line_min_percentile)
        legend_variable_names.append(f"Percentiel {percentile} - {percentile_max}")

    if plot_mean:
        mean_line = []
        for x in unique_x:
            mean_line.append(np.mean(data_y[data[x_name] == x]))
        line_mean, = plt.plot(unique_x, mean_line, color='black')

        legend_variables.append(line_mean)
        legend_variable_names.append("Gemiddelde")

    if x_name == "time_step":
        restrictions = [("paths", 1000, 5000), ("paths", 6000, 10000), ("paths", 11000, 15000), ("paths", 16000, 20000)]
    else:
        restrictions = [("time_step", 5, 50), ("time_step", 55, 100), ("time_step", 200, 600), ("time_step", 700, 1000)]
    # todo deze stuk code verwijderen
    for restrict in restrictions:
        scat_points = plt.scatter(data_x[(restrict[1] <= data[restrict[0]]) & (data[restrict[0]] <= restrict[2])],
                                  data_y[(restrict[1] <= data[restrict[0]]) & (data[restrict[0]] <= restrict[2])])

        dict_rename = {"time_step": "#stappen",
                       "paths": "#paden"}

        legend_variables.append(scat_points)
        legend_variable_names.append(f"{dict_rename[restrict[0]]}: [{restrict[1]}-{restrict[2]}]")

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
    # x_name = "time_step"
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
                data_options = read_data(model, option)

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
                                         title=title_plot.format(dict_option_names_nl[option], dict_model_names[model]),
                                         xlabel=x_label,
                                         ylabel=y_label,
                                         plot_mean=plot_mean,
                                         plot_min_max=plot_min_max,
                                         plot_percentile=plot_percentile,
                                         percentile=percentile,
                                         restriction=restriction)
