import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv


def read_data(filename):
    data = pd.read_csv(filename, header=0, comment='#')
    return data


filename = 'Test - steps and paths accuracy -version 2-1.csv'
data = read_data(filename)

number_paths = data['paths'].unique()
lst_legend = []
for path in number_paths:
    time_steps = data['time_step'][data['paths'] == path]
    # accuracy_abs = data['accuracy_absolute'][data['paths'] == path]
    # p, = plt.plot(time_steps, accuracy_abs)
    accuracy = data['accuracy_normal'][data['paths'] == path]
    p, = plt.plot(time_steps, accuracy)
    # lst_legend.append(p)
# plt.legend(lst_legend, number_paths)
plt.show()

# plt.plot(data["time_step"], data["accuracy_normal"])
# plt.show()
