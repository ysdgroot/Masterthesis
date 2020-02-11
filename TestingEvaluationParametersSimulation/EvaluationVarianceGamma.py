import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from sklearn.linear_model import LinearRegression


def read_data(filename):
    data = pd.read_csv(filename, header=0, comment='#')
    return data


filename = 'Test-steps and accuracy-VG-v1.csv'
data = read_data(filename)

number_paths = data['paths'].unique()
step_sizes = data['time_step'].unique()
show_plots = False

print(number_paths)

variance_global_paths = []
for path in number_paths:
    variance_global_paths.append(np.var(data["option_value"][data["paths"] == path]))

print(variance_global_paths)

print(step_sizes)
variance_global_step = []
for step_size in step_sizes:
    variance_global_step.append(np.var(data["option_value"][data["time_step"] == step_size]))

print(variance_global_step)

variance_matrix = []
for path in number_paths:
    variance_paths = list()
    for step_size in step_sizes:
        steps = data["time_step"] == step_size
        pad = data["paths"] == path
        variance_paths.append(np.var(data["option_value"][steps & pad]))
    variance_matrix.append(variance_paths)

for row in variance_matrix:
    print(row)
