from ModelsStock.VarianceGamma import VarianceGamma
from OptionModels.PlainVanilla import PlainVanilla
import time
import csv
import math
import numpy as np

# Testing paths
time_steps_per_maturities = [i for i in range(100, 1001, 100)]
amount_paths = [i for i in range(15000, 20001, 1000)]
add_header = False
write_comment_info = False

# todo instellen van andere parameters
file_name = 'Test-steps and accuracy-VG-v1.csv'
maturity = 10
interest_rate = 0.001
sigma = 0.25
nu = 0.75
theta = -0.2
start_price = 100
strike_price = 100

VG = VarianceGamma(interest_rate, theta, sigma, nu)
option = PlainVanilla()
